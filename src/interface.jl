# User interface: plans are AbstractFFTs plans (`p * x`, `mul!`, `inv`, `\`). `fft`, `plan_fft`
# and the other functions here are FFTL's own, exported under FFTW.jl's names; the methods that
# make AbstractFFTs's generic functions reach FFTL are in src/abstractffts.jl.

const FFTLFloat = Union{Float32,Float64}

# The thread default for new plans: set_num_threads(n) sets it (the drop-in for
# FFTW.set_num_threads); 0 means Threads.nthreads().
const DEFAULT_THREADS = Ref(0)
plan_threads() = DEFAULT_THREADS[] == 0 ? Threads.nthreads() : DEFAULT_THREADS[]
"""
    FFTL.set_num_threads(n)

Default number of threads for the plans made afterwards (`threads` keyword not given), as
`FFTW.set_num_threads`. Without a call, plans use `Threads.nthreads()`.
"""
set_num_threads(n::Integer) = (DEFAULT_THREADS[] = Int(n); nothing)

# The planning flags of FFTW.jl's interface, accepted and ignored: an FFTL plan does not
# depend on measurements made while planning, so there is nothing for them to select.
"""
    FFTL.ESTIMATE, FFTL.MEASURE, FFTL.PATIENT, FFTL.EXHAUSTIVE

Values for the `flags` keyword of the plan functions, accepted for FFTW.jl compatibility.
They have no effect: an FFTL plan does not depend on measurements made while planning.
"""
const ESTIMATE = UInt32(1 << 0)
const MEASURE = UInt32(1 << 1)
const PATIENT = UInt32(1 << 2)
const EXHAUSTIVE = UInt32(1 << 3)

# The FFTW-style keywords: num_threads maps to threads, flags and timelimit are accepted and
# ignored (documented); the remaining keywords go to the plan constructors.
planthreads(threads, num_threads) =
  num_threads !== nothing ? Int(num_threads) : threads !== nothing ? Int(threads) : plan_threads()

# The transformed dimensions: sorted, without duplicates, each within 1:N (an integer is one
# dimension). The message is a literal: the run-time plans are compiled ahead of time.
function normregion(region, N::Int)
  reg = sort!(unique!(collect(Int, region isa Integer ? (region,) : region)))
  all(d -> 1 <= d <= N, reg) || throw(ArgumentError("region is not within the array dimensions"))
  reg
end

"""
    FFTLPlan{T,N}

Plan for the complex DFT of `N`-dimensional arrays of size `sz` along the dimensions `region`,
with sign `σ` (-1 forward, +1 backward, unnormalised). The dimensions are transformed one after
the other, each by a kernel that handles all lanes of the dimension (see kernels.jl).
"""
mutable struct FFTLPlan{T<:FFTLFloat,N,D,Ks<:Tuple} <: AbstractFFTs.Plan{Complex{T}}
  sz::NTuple{N,Int}
  region::NTuple{D,Int}
  σ::Int
  kernels::Ks
  work1::Vector{Complex{T}}
  work2::Vector{Complex{T}}
  nthreads::Int
  tbuf::Vector{Vector{Complex{T}}}     # two work buffers per thread
  pinv::AbstractFFTs.Plan{Complex{T}}
  FFTLPlan{T,N,D,Ks}(sz, region, σ, kernels, w1, w2, nt, tbuf) where {T,N,D,Ks} =
    new{T,N,D,Ks}(sz, region, σ, kernels, w1, w2, nt, tbuf)
end

function FFTLPlan{T}(sz::NTuple{N,Int}, region, σ::Int; threads::Integer = Threads.nthreads()) where {T<:FFTLFloat,N}
  reg = Tuple(normregion(region, N))
  nt = check_plan(sz, σ, threads)
  # dimensions with the same length and stride class share the kernel; a single long contiguous
  # transform on several threads uses the four-step algorithm
  cache = Dict{Tuple{Int,Int},Kernel{T}}()
  function dim_kernel(d)
    fourstep_dim(sz, d, nt) && return FourStep{T}(sz[d], σ, nt)
    composite_mt_dim(T, sz, d, nt) && return CompositeMT{T}(sz[d], σ, nt)
    bluestein_mt_dim(sz, d, nt) && return BluesteinMT{T}(sz[d], σ, nt)
    S, _ = dim_extent(sz, d)
    lines_mt_dim(sz, d, nt) && return LinesMT{T}(sz[d], σ, S, nt)
    get!(() -> kernel(T, sz[d], σ, S), cache, (sz[d], S > LARGE_STRIDE || !smooth(sz[d]) ? S : 0))
  end
  kernels = Tuple(dim_kernel(d) for d in reg)
  nwork, ntwork = work_sizes(sz, reg, nt)
  tbuf = thread_buffers(T, ntwork, 2nt)
  FFTLPlan{T,N,length(reg),typeof(kernels)}(sz, reg, σ, kernels, Vector{Complex{T}}(undef, nwork),
                                            Vector{Complex{T}}(undef, nwork), nt, tbuf)
end

Base.size(p::FFTLPlan) = p.sz
Base.ndims(::FFTLPlan{T,N}) where {T,N} = N
AbstractFFTs.fftdims(p::FFTLPlan) = p.region
AbstractFFTs.AdjointStyle(::FFTLPlan) = AbstractFFTs.FFTAdjointStyle()

function Base.show(io::IO, p::FFTLPlan{T}) where {T}
  dir = p.σ == -1 ? "forward" : "backward"
  print(io, "FFTL $dir plan for $(join(p.sz, "×")) Complex{$T}, dims $(p.region)")
end

function Base.show(io::IO, ::MIME"text/plain", p::FFTLPlan{T}) where {T}
  show(io, p)
  for (d, k) in zip(p.region, p.kernels)
    print(io, "\n  dim $d: ")
    k isa Chain && print(io, "radices ", radix_tuple(k))
    k isa Bluestein && print(io, "Bluestein, convolution length ", k.fwd.n)
    k isa Composite && print(io, "chain of length ", k.s, ", Bluestein of length ", k.q, ", convolution length ", k.blue.fwd.n)
    k isa CompositeMT && print(io, "chain of length ", k.s, ", Bluestein of length ", k.q, " on threads, convolution length ", k.blues[1].fwd.n)
    k isa BluesteinMT && print(io, "Bluestein on threads, convolution length ", k.fwd.n)
    k isa LinesMT && print(io, "one kernel per thread, blocks of ", k.B, " lines")
    k isa FourStep && print(io, "four-step $(k.n1) × $(k.n2), radices ", radix_tuple(k.k1), " and ", radix_tuple(k.k2))
  end
end
radix_tuple(::Chain{T,Rs}) where {T,Rs} = Rs

check_size(p, x) = size(x) == p.sz ||
  throw(DimensionMismatch("plan for size $(p.sz) applied to an array of size $(size(x))"))

# Dimensions one after the other; the tuple of kernels is peeled so that each call is static.
# Dimensions 1 and 2 are fused per plane when a plane fits in cache (`run_planes!`).
function apply_dims!(y::Array{Complex{T}}, x::Array{Complex{T}}, p::FFTLPlan{T}, ks::Tuple, ds::Tuple) where {T}
  isempty(ks) && return y
  if length(ks) >= 2 && ds[1] == 1 && ds[2] == 2 && ks[1] isa Chain && ks[2] isa Chain
    n1, n2 = p.sz[1], p.sz[2]
    outer = dim_extent(p.sz, 2)[2]
    if fuse_planes(n1, n2, outer, p.nthreads)
      GC.@preserve x y p run_planes!(cptr(y), cptr(x), p, ks[1], ks[2], n1, n2, outer)
      return apply_dims!(y, y, p, Base.tail(Base.tail(ks)), Base.tail(Base.tail(ds)))
    end
  end
  k, d = first(ks), first(ds)
  S, outer = dim_extent(p.sz, d)
  n = p.sz[d]
  GC.@preserve x y p run_dim!(cptr(y), cptr(x), p, k, S, n, outer)
  apply_dims!(y, y, p, Base.tail(ks), Base.tail(ds))
end

function LinearAlgebra.mul!(y::Array{Complex{T},N}, p::FFTLPlan{T,N}, x::Array{Complex{T},N}) where {T,N}
  check_size(p, x); check_size(p, y)
  isempty(p.region) && return copyto!(y, x)
  apply_dims!(y, x, p, p.kernels, p.region)
end

function LinearAlgebra.mul!(y::AbstractArray{Complex{T},N}, p::FFTLPlan{T,N}, x::AbstractArray{Complex{T},N}) where {T,N}
  check_size(p, x); check_size(p, y)
  copyto!(y, mul!(Array{Complex{T},N}(undef, p.sz), p, Array{Complex{T},N}(x)))
end

Base.:*(p::FFTLPlan{T,N}, x::AbstractArray{<:Number,N}) where {T,N} =
  mul!(Array{Complex{T},N}(undef, p.sz), p, convert(Array{Complex{T},N}, x))

AbstractFFTs.plan_inv(p::FFTLPlan{T}) where {T} =
  AbstractFFTs.ScaledPlan(FFTLPlan{T}(p.sz, p.region, -p.σ; threads = p.nthreads),
                          AbstractFFTs.normalization(T, p.sz, p.region))

"""
    plan_fft(x::AbstractArray, region = 1:ndims(x); threads, num_threads, flags, timelimit)
    plan_bfft(x, region = 1:ndims(x); …), plan_ifft(x, region = 1:ndims(x); …)

Plans for the forward, unnormalised backward and normalised inverse DFT along the dimensions
`region` of arrays like `x`. Float32 and Float64 (real or complex) use the SIMD kernels, and
transforms large enough to profit run on up to `threads` threads (default: the value of
[`FFTL.set_num_threads`](@ref), else `Threads.nthreads()`; `num_threads` is a synonym).
`flags` and `timelimit` are accepted for FFTW.jl compatibility and have no effect: FFTL does
not measure while planning. Other floating-point types (BigFloat, Double64, Float128, …) are
transformed in their own precision by a generic, single-threaded path; Float16, integers and
other numbers are converted to ComplexF64.
"""
function plan_fft(x::AbstractArray, region = 1:ndims(x); threads = nothing, num_threads = nothing,
                  flags = nothing, timelimit = nothing, kw...)
  generic_float(elreal(x)) && return GPlan{elreal(x)}(size(x), region, -1)
  FFTLPlan{plan_type(x)}(size(x), region, -1; threads = planthreads(threads, num_threads), kw...)
end
function plan_bfft(x::AbstractArray, region = 1:ndims(x); threads = nothing, num_threads = nothing,
                   flags = nothing, timelimit = nothing, kw...)
  generic_float(elreal(x)) && return GPlan{elreal(x)}(size(x), region, 1)
  FFTLPlan{plan_type(x)}(size(x), region, 1; threads = planthreads(threads, num_threads), kw...)
end
# the normalisation over the plan's dimensions (the region without duplicates)
function plan_ifft(x::AbstractArray, region = 1:ndims(x); kw...)
  p = plan_bfft(x, region; kw...)
  AbstractFFTs.ScaledPlan(p, AbstractFFTs.normalization(generic_float(elreal(x)) ? elreal(x) : plan_type(x), size(x), AbstractFFTs.fftdims(p)))
end

elreal(x::AbstractArray) = real(eltype(x))

plan_type(::AbstractArray{Complex{T}}) where {T<:FFTLFloat} = T
plan_type(::AbstractArray{T}) where {T<:FFTLFloat} = T
plan_type(::AbstractArray) = Float64

"""
    fft(x, region = 1:ndims(x))

Forward DFT of `x` along `region` (all dimensions by default), unnormalised. Equal to
`plan_fft(x, region) * x`.
"""
fft(x::AbstractArray, region = 1:ndims(x)) = plan_fft(x, region) * x

"""
    bfft(x, region = 1:ndims(x))

Backward DFT along `region`, unnormalised: applying `bfft` to `fft(x)` multiplies `x` by the
number of transformed elements.
"""
bfft(x::AbstractArray, region = 1:ndims(x)) = plan_bfft(x, region) * x

"""
    ifft(x, region = 1:ndims(x))

Normalised inverse DFT along `region`: `ifft(fft(x)) ≈ x`.
"""
ifft(x::AbstractArray, region = 1:ndims(x)) = plan_ifft(x, region) * x

"""
    InplacePlan(p)

A plan that applies `p` in place: `q * x` overwrites `x` and returns it, as FFTW.jl's in-place
plans do. Returned by `plan_fft!`, `plan_bfft!`, `plan_ifft!`, `plan_dct!` and `plan_idct!`.
"""
mutable struct InplacePlan{T,P<:AbstractFFTs.Plan{T}} <: AbstractFFTs.Plan{T}
  p::P
  pinv::AbstractFFTs.Plan{T}
  InplacePlan(p::AbstractFFTs.Plan{T}) where {T} = new{T,typeof(p)}(p)
end

Base.size(q::InplacePlan) = size(q.p)
Base.ndims(q::InplacePlan) = ndims(q.p)
AbstractFFTs.fftdims(q::InplacePlan) = AbstractFFTs.fftdims(q.p)
AbstractFFTs.plan_inv(q::InplacePlan) = InplacePlan(inv(q.p))
LinearAlgebra.mul!(y::AbstractArray, q::InplacePlan, x::AbstractArray) = mul!(y, q.p, x)
Base.:*(q::InplacePlan, x::AbstractArray) = inplace!(q.p, x)
Base.show(io::IO, q::InplacePlan) = (print(io, "in-place "); show(io, q.p))

# The complex CPU plans may run with y === x: every pass loads a line before it stores it.
# Other plans go through a temporary.
inplace!(p::Union{FFTLPlan{T},AbstractFFTs.ScaledPlan{<:Any,<:FFTLPlan{T}}}, x::Array{Complex{T}}) where {T} = mul!(x, p, x)
inplace!(p::AbstractFFTs.Plan, x::AbstractArray) = copyto!(x, p * x)

"""
    plan_fft!(x, region = 1:ndims(x); …), plan_bfft!, plan_ifft!
    fft!(x, region = 1:ndims(x)), bfft!, ifft!

In-place forms: the plans overwrite the array they are applied to with `p * x`, and `fft!`,
`bfft!` and `ifft!` overwrite `x`. Keywords as for [`plan_fft`](@ref).
"""
plan_fft!(x::AbstractArray{<:Complex}, region = 1:ndims(x); kw...) = InplacePlan(plan_fft(x, region; kw...))
plan_bfft!(x::AbstractArray{<:Complex}, region = 1:ndims(x); kw...) = InplacePlan(plan_bfft(x, region; kw...))
plan_ifft!(x::AbstractArray{<:Complex}, region = 1:ndims(x); kw...) = InplacePlan(plan_ifft(x, region; kw...))
fft!(x::AbstractArray{<:Complex}, region = 1:ndims(x)) = plan_fft!(x, region) * x
bfft!(x::AbstractArray{<:Complex}, region = 1:ndims(x)) = plan_bfft!(x, region) * x
ifft!(x::AbstractArray{<:Complex}, region = 1:ndims(x)) = plan_ifft!(x, region) * x
