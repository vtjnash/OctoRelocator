# This file is a part of Julia. License is MIT: http://julialang.org/license
#__precompile__()
module OctoRelocator

import Base: GMP, Bottom, svec, unsafe_convert, uncompressed_ast
import Base: get!, getindex, setindex!
import Core: arrayref, arrayset

# a much faster ObjectIdDict, but which does not guarantee object uniqueness for isbits objects
type FuzzyDict
    dict::Dict{UInt, Int}
    FuzzyDict() = new(Dict{UInt, Int}())
end
get!(d::FuzzyDict, k::ANY, v) = get!(d.dict, UInt(pointer_from_objref(k)), v)
getindex(d::FuzzyDict, k::ANY) = getindex(d.dict, UInt(pointer_from_objref(k)))
setindex!(d::FuzzyDict, v, k::ANY) = setindex!(d.dict, v, UInt(pointer_from_objref(k)))
typealias ObjectIdDict FuzzyDict

if !isdefined(Base,:unsafe_write)
unsafe_write(s::IO, p::Ptr{UInt8}, nb::UInt) = write(s, p, Int(nb))
end

export serialize, deserialize

## serializing values ##

type FastSerializationState{I<:IO}
    io::I
    hcounter::Int
    ocounter::Int
    serialized::ObjectIdDict
    workq::Vector{Any}
    relocarrayid::Vector{Int}
    relocarrays::Vector{Vector{Int}}
    FastSerializationState(io::I) = new(io, 0, 0, ObjectIdDict(), Vector{Any}(), Vector{Int}(), Vector{Vector{Int}}())
end
FastSerializationState(io::IO) = FastSerializationState{typeof(io)}(io)
typealias SerializationState FastSerializationState

function serialize(s::IO, x)
    fastserialize(FastSerializationState(s), x)
    return s
end
serialize(x) = serialize(PipeBuffer(), x)

const TAGS = Any[
    Symbol, Int8, UInt8, Int16, UInt16, Int32, UInt32,
    Int64, UInt64, Int128, UInt128, Float32, Float64, Char, Ptr,
    DataType, Union, Function,
    Tuple, Array, Expr,
    #LongSymbol, LongTuple, LongExpr,
    Symbol, Tuple, Expr,  # dummy entries, intentionally shadowed by earlier ones
    LineNumberNode, SymbolNode, LabelNode, GotoNode,
    QuoteNode, TopNode, TypeVar, Box, LambdaStaticData,
    Module, #=UndefRefTag=#Symbol, Task, ASCIIString, UTF8String,
    UTF16String, UTF32String, Float16,
    SimpleVector, #=Any Value=#Symbol, :reserved11, :reserved12,

    (), Bool, Any, :Any, Bottom, :reserved21, :reserved22, Type,
    :Array, :TypeVar, :Box, :Tuple,
    :lambda, :body, :return, :call, symbol("::"),
    :(=), :null, :gotoifnot, :A, :B, :C, :M, :N, :T, :S, :X, :Y,
    :a, :b, :c, :d, :e, :f, :g, :h, :i, :j, :k, :l, :m, :n, :o,
    :p, :q, :r, :s, :t, :u, :v, :w, :x, :y, :z,
    :add_int, :sub_int, :mul_int, :add_float, :sub_float,
    :mul_float, :unbox, :box,
    :eq_int, :slt_int, :sle_int, :ne_int,
    :arrayset, :arrayref,
    :Core, :Base, svec(), Tuple{},
    :reserved17, :reserved18, :reserved19,
    false, true, nothing, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32
]

const ser_version = 3 # do not make changes without bumping the version #!

const NTAGS = length(TAGS)

@inline function sertag(v::ANY)
    ptr = pointer_from_objref(v)
    ptags = convert(Ptr{Ptr{Void}}, pointer(TAGS))
    @inbounds for i = 1:NTAGS
        ptr == unsafe_load(ptags,i) && return (i+1)%Int32
    end
    return Int32(-1)
end
desertag(i::Int32) = TAGS[i-1]

# tags >= this just represent themselves, their whole representation is 1 byte
const VALUE_TAGS = sertag(())
const ZERO_TAG = sertag(0)
const TRUE_TAG = sertag(true)
const FALSE_TAG = sertag(false)
const EMPTYTUPLE_TAG = sertag(())
const TUPLE_TAG = sertag(Tuple)
const LONGTUPLE_TAG = Int32(sertag(Expr)+2)
const SIMPLEVECTOR_TAG = sertag(SimpleVector)
const SYMBOL_TAG = sertag(Symbol)
const LONGSYMBOL_TAG = Int32(sertag(Expr)+1)
const ARRAY_TAG = sertag(Array)
const UNDEFREF_TAG = Int32(sertag(Module)+1)
const VALUE_TAG = Int32(sertag(SimpleVector)+1)
const EXPR_TAG = sertag(Expr)
const LONGEXPR_TAG = Int32(sertag(Expr)+3)
const MODULE_TAG = sertag(Module)
const FUNCTION_TAG = sertag(Function)
const LAMBDASTATICDATA_TAG = sertag(LambdaStaticData)
const TASK_TAG = sertag(Task)
const DATATYPE_TAG = sertag(DataType)
const INT_TAG = sertag(Int)
const UNION_TAG = sertag(Union)

writetag(s::IO, tag) = (write(s, UInt8(tag)); nothing)

function write_as_tag(s::IO, tag)
    tag < VALUE_TAGS && write(s, UInt8(0))
    write(s, UInt8(tag))
    nothing
end

macro writefield(s, x, xptr, t, i, fldty) # precondition: isbits(fldty::DataType)
    return esc(quote unsafe_write(s.io, convert(Ptr{UInt8}, xptr + Base.field_offset(t, i)), UInt(fldty.size)) end)
    #ccall(:jl_set_nth_field, Void, (Any, Csize_t, Any), x, i-1, read(s.io, fldty))
end

function readfield{T}(::Type{T}, s::IO, fptr)
    v = read(s, T)
    unsafe_store!(convert(Ptr{T}, fptr), v)
    nothing
end

macro readfield(s, x, xptr, t, i, fldty) # precondition: isbits(fldty::DataType)
    return esc(quote
        fldsz = $fldty.size
        if fldsz == 1
            readfield(UInt8, s.io, xptr + Base.field_offset(t, i))
        elseif fldsz == 2
            readfield(UInt16, s.io, xptr + Base.field_offset(t, i))
        elseif fldsz == 4
            readfield(UInt32, s.io, xptr + Base.field_offset(t, i))
        elseif fldsz == 8
            readfield(UInt64, s.io, xptr + Base.field_offset(t, i))
        else
            readfield($fldty, s.io, xptr + Base.field_offset(t, i))
        end
        #ccall(:jl_set_nth_field, Void, (Any, Csize_t, Any), x, i-1, read(s.io, fldty))
    end)
end

LLT_ALIGN(v::Unsigned, typ::DataType) = (v + Base.type_alignment(typ) - 1) & ~UInt(Base.type_alignment(typ) - 1)
const fielddesc_type_offset = LLT_ALIGN(Base.field_offset(DataType,
        findfirst(Base.fieldnames(DataType), :ninitialized) - 1) +
    sizeof(fieldtype(DataType, :ninitialized)), UInt32) + sizeof(UInt32)
const fielddesc_offset = LLT_ALIGN(
    LLT_ALIGN(fielddesc_type_offset + sizeof(UInt32), Ptr{Void}) + 2 * sizeof(Ptr{Void}),
    DataType)
function jl_field_isptr(t::DataType, i::Int) # 1 <= i <= nfields(t)
    tptr = pointer_from_objref(t)
    fielddesc_type = (unsafe_load(convert(Ptr{UInt32}, tptr) + fielddesc_type_offset) & 0xd0000000) >> 30
    i = 2 * i
    if fielddesc_type == 0
        isptr = (unsafe_load(convert(Ptr{UInt8}, tptr) + fielddesc_offset, i) >> 7) != 0
    elseif fielddesc_type == 1
        isptr = (unsafe_load(convert(Ptr{UInt16}, tptr) + fielddesc_offset, i) >> 15) != 0
    else
        isptr = (unsafe_load(convert(Ptr{UInt32}, tptr) + fielddesc_offset, i) >> 31) != 0
    end
    #fldty = t.types[i]
    #return typeof(fldty) === DataType && isbits(fldty::DataType)
end

@inline function fastserialize_any(s::SerializationState, obj::ANY)
    t = typeof(obj)
    if is(t, DataType) || is(t, Symbol) || is(t, Module) || is(t, Union)
        return serialize_header_obj(s, obj)
    end
    xid = get!(s.serialized, obj, s.ocounter)::Int
    if xid == s.ocounter
        s.ocounter -= 1
        push!(s.workq, obj)
    end
    return xid
end

function fastserialize(s::SerializationState, obj::ANY)
    workq = s.workq
    xid = 1
    s.hcounter = 1
    s.ocounter = -1
    id = fastserialize_any(s, obj)
    relocations = Vector{Tuple{Int,Int,Int}}()
    while xid <= length(workq)
        x = workq[xid]
        tag = sertag(x)
        if tag <= 0
            t = typeof(x)::DataType
            @assert !(is(t, DataType) || is(t, Symbol) || is(t, Module) || is(t, Union))
            if is(t.name, Array.name)
                serialize_array(s, x, xid)
            elseif is(t, SimpleVector)
                #serialize_sv(s, x::SimpleVector, xid)
                sv = x::SimpleVector
                for i in 1:length(sv)
                    id = fastserialize_any(sv[i])::Int
                    push!(relocations, (xid, i, id))
                end
            else
                t_tag = sertag(t)
                t_tag <= 0 && serialize_header_obj(s, t)
                nf = nfields(t)
                for i in 1:nf
                    if jl_field_isptr(t, i)
                        if isdefined(x, i)
                            xf = getfield(x, i)
                            id = fastserialize_any(s, xf)
                            push!(relocations, (xid, i, id))
                        end
                    end
                end
            end
        end
        xid += 1
    end
    writetag(s.io, UNDEFREF_TAG) # mark the end of the header

    for xid = 1:length(workq)
        x = workq[xid]
        tag = sertag(x)
        if tag > 0
            write_as_tag(s.io, tag)
        else
            t = typeof(x)::DataType
            @assert !(is(t, DataType) || is(t, Symbol) || is(t, Module) || is(t, Union))
            if is(t.name, Array.name)
                serialize_array(s, x::Array, 0)
            elseif is(t, SimpleVector)
                #serialize_sv(s, x::SimpleVector, 0)
                sv = x::SimpleVector
                writetag(s.io, SIMPLEVECTOR_TAG)
                l = length(sv)::Int
                write(s.io, l)
            elseif isa(x, Int)
                serialize_int(s, x::Int)
            else
                t_tag = sertag(t)
                if t_tag > 0
                    writetag(s.io, t_tag)
                else
                    writetag(s.io, VALUE_TAG)
                    write(s.io, s.serialized[t]::Int)
                end

                nf = nfields(t)
                xptr = pointer_from_objref(x)
                for i in 1:nf
                    if !jl_field_isptr(t, i)
                        fldty = t.types[i]::DataType
                        @writefield(s, x, xptr, t, i, fldty)
                    end
                end
            end
        end
    end

    # write all of the pending relocations
    writetag(s.io, UNDEFREF_TAG)
    write(s.io, Int(length(relocations)))
    write(s.io, relocations)
    write(s.io, length(s.relocarrays))
    for i = 1:length(s.relocarrays)
        write(s.io, s.relocarrayid[i])
        write(s.io, s.relocarrays[i])
    end
    write(s.io, s.serialized[obj]::Int)
    nothing
end

# the header contains objects that cannot be recursive,
# esp. all types that might be needed later
function serialize_header_obj(s::SerializationState, x::ANY)
    xid = get!(s.serialized, x, 0)::Int
    xid > 0 && return xid

    tag = sertag(x)
    if tag > 0
        write_as_tag(s.io, tag)
    else
        t = typeof(x)::DataType
        if isa(x, DataType)
            serialize_datatype(s, x)
        elseif isa(x, Module)
            serialize_module(s, x)
        elseif isa(x, Symbol)
            serialize_symbol(s, x)
        elseif isa(x, Int)
            serialize_int(s, x)
        elseif isa(x, Union)
            serialize_union(s, x)
        else
            t_tag = sertag(t)
            if t_tag > 0
                writetag(s.io, t_tag)
            else
                tid = serialize_header_obj(s, t)
                writetag(s.io, VALUE_TAG)
                write(s.io, tid)
            end
            nf = nfields(t)
            if nf == 0 && t.size > 0 # bitstype
                write(s.io, x)
            else # other isbits
                @assert isbits(t)
                xptr = pointer_from_objref(x)
                for i = 1:nf
                    fldty = t.types[i]::DataType
                    @writefield(s, x, xptr, t, i, fldty)
                end
            end
        end
    end

    xid = s.hcounter
    s.hcounter += 1
    s.serialized[x] = xid
    return xid
end

## HEADER OBJECTS ##################################

function serialize_mod_names(s::SerializationState, m::Module, writeout::Bool)
    p = module_parent(m)
    if m !== p
        serialize_mod_names(s, p, writeout)
        name = module_name(m)
        if writeout
            write(s.io, s.serialized[name]::Int)
        else
            serialize_header_obj(s, name)
        end
    end
    nothing
end

function serialize_module(s::SerializationState, m::Module)
    serialize_mod_names(s, m, false)
    writetag(s.io, MODULE_TAG)
    serialize_mod_names(s, m, true)
    write(s.io, 0)
    nothing
end

function serialize_datatype(s::SerializationState, t::DataType)
    tname = t.name.name
    id_tname = serialize_header_obj(s, tname)
    mod = t.name.module
    id_mod = serialize_header_obj(s, mod)
    count = -1
    tp = t.parameters
    if !isempty(tp)
        if isdefined(mod, tname) && is(t, getfield(mod, tname))
            count = 0
        else
            count = length(tp)
            for i = 1:count
                serialize_header_obj(s, tp[i])
            end
        end
    end
    writetag(s.io, DATATYPE_TAG)
    write(s.io, id_tname)
    write(s.io, id_mod)
    if count >= 0
        write(s.io, count)
        for i = 1:count
            write(s.io, s.serialized[tp[i]]::Int)
        end
    end
    nothing
end

function serialize_int(s::SerializationState, n::Int)
    if 0 <= n <= 32
        write(s.io, UInt8(ZERO_TAG+n))
        return
    end
    writetag(s.io, INT_TAG)
    write(s.io, n)
    nothing
end

function serialize_union(s::SerializationState, u::Union)
    ut = u.types
    count = length(ut)::Int
    for i = 1:count
        serialize_header_obj(s, ut[i])
    end
    writetag(s.io, UNION_TAG)
    write(s.io, count)
    for i = 1:count
        serialize_header_obj(s, s.serialized[ut[i]]::Int)
    end
    nothing
end

function serialize_symbol(s::SerializationState, x::Symbol)
    pname = unsafe_convert(Ptr{UInt8}, x)
    ln = UInt(ccall(:strlen, Csize_t, (Ptr{UInt8},), pname))
    if ln <= 255
        writetag(s.io, SYMBOL_TAG)
        write(s.io, UInt8(ln))
    else
        writetag(s.io, LONGSYMBOL_TAG)
        write(s.io, Int32(ln))
    end
    unsafe_write(s.io, pname, ln)
    nothing
end

## MISC OBJECT SERIALIZERS ##################################

function serialize_array_bits(s::IO, a)
    elty = eltype(a)
    if elty === Bool && !isempty(a)
        last = a[1]
        count = 1
        for i = 2:length(a)
            if a[i] != last || count == 127
                write(s, UInt8((UInt8(last)<<7) | count))
                last = a[i]
                count = 1
            else
                count += 1
            end
        end
        write(s, UInt8((UInt8(last)<<7) | count))
    else
        write(s, a)
    end
    nothing
end

function serialize_array(s::SerializationState, a::Array, xid::Int)
    if xid != 0
        elty = eltype(a)
        serialize_header_obj(s, elty)
        if typeof(elty) !== DataType || !isbits(elty::DataType)
            na = length(a)::Int
            reloca = Array{Int}(na)
            for fld = 1:na
                if isdefined(a, fld)
                    id = fastserialize_any(s, a[fld])::Int
                else
                    id = 0
                end
                reloca[fld] = id
            end
            push!(s.relocarrayid, xid)
            push!(s.relocarrays, reloca)
        end
    else
        elty = eltype(a)
        writetag(s.io, ARRAY_TAG)
        write(s.io, s.serialized[elty]::Int)
        nd = ndims(a)::Int
        write(s.io, nd)
        for i = 1:nd
            write(s.io, size(a, i))
        end
        if typeof(elty) === DataType && isbits(elty::DataType)
            serialize_array_bits(s.io, a)
        end
    end
    nothing
end

function serialize_sv(s::SerializationState, v::SimpleVector, xid::Int)
    if xid != 0
        for x in v
            id = fastserialize_any(s, x)
        end
        nothing
    end
end

#
#
#function fastserialize_any(s::SerializationState, obj::ANY)
#    xid::Int = get!(s.serialized, obj, 0)::Int
#    if xid != 0
#        writetag(s.io, UNDEFREF_TAG)
#        write(s.io, xid)
#        return
#    end
#    relocations = Vector{Any}()
#    relocarrayid = Vector{Int}()
#    relocarrays = Vector{Vector{Int}}()
#    push!(relocations, obj)
#    fld::Int = 1
#    while !isempty(relocations)
#        x = relocations[1]
#        xid = s.serialized[x]::Int
#        reloc::Tuple{Int,Int,Int} = (0, 0, 0)
#        if xid == 0
#            shift!(relocations)
#        else
#            # already serialized x, now look at its fields
#            if isa(x, Array)
#                x = x::Array
#                na::Int = length(x)
#                while fld <= na
#                    if isdefined(x, fld)
#                        xf = arrayref(x, fld)
#                        id = get!(s.serialized, xf, 0)::Int
#                        if id == 0
#                            # need to serialize this value
#                            x = xf
#                            break
#                        end
#                    end
#                    fld += 1
#                end
#                if fld > na
#                    # finished with serializing this array
#                    reloca = Array{Int}(na)
#                    for fld = 1:na
#                        if isdefined(x, fld)
#                            id = s.serialized[arrayref(x, fld)]::Int
#                        else
#                            id = 0
#                        end
#                        reloca[fld] = id
#                    end
#                    push!(relocarrayid, xid)
#                    push!(relocarrays, reloca)
#                    fld = 1
#                    xid = 0
#                    shift!(relocations)
#                    continue
#                end
#                fld += 1
#            else
#                t = typeof(x)::DataType
#                nf::Int = nfields(t)
#                while fld <= nf
#                    fldty = t.types[i]
#                    if typeof(fldty) !== DataType || !isbits(fldty::DataType)
#                        if isdefined(x, fld)
#                            xf = getfield(x, fld)
#                            id = s.serialized[xf]::Int
#                            if id == 0
#                                # need to serialize this value
#                                reloc = (xid, fld, s.counter)
#                                x = xf
#                                break
#                            elseif id > xid
#                                # already got serialized, just record the relocation
#                                push!(s.relocations, (xid, fld, id))
#                            end
#                        end
#                    end
#                    fld += 1
#                end
#                if fld > nf
#                    # finished with serializing this object
#                    fld = 1
#                    xid = 0
#                    shift!(relocations)
#                    continue
#                end
#                fld += 1
#            end
#        end
#
#        # serialize x
#        xid = s.counter
#        s.counter += 1
#        s.serialized[x] = xid
#        tag = sertag(x)
#        if tag > 0
#            write_as_tag(s.io, tag)
#        else
#            t = typeof(x)::DataType
#            #if isa(x, Tuple)
#            #    serialize_tuple(s, x)
#            if isa(x, Array)
#                serialize_array(s, x, relocations)
#            elseif isa(x, DataType)
#                serialize_datatype(s, x)
#            elseif isa(x, Symbol)
#                serialize_symbol(s, x)
#            elseif isa(x, Module)
#                serialize_module(s, x)
#            elseif isa(x, Expr)
#                serialize_expr(s, x)
#            elseif isa(x, SimpleVector)
#                serialize_sv(s, x)
#            elseif isa(x, Int)
#                serialize_int(s, x)
#            elseif isa(x, Union)
#                serialize_union(s, x)
#            else
#                t_tag = sertag(t)
#                if t_tag > 0
#                    writetag(s.io, t_tag)
#                else
#                    writetag(s.io, VALUE_TAG)
#                    fastserialize_any(s, t)
#                end
#
#                nf = nfields(t)
#                if nf == 0 && t.size > 0 # bitstype
#                    write(s.io, x)
#                else
#                    needsreloc = false
#                    xptr = pointer_from_objref(x)
#                    for i in 1:nf
#                        fldty = t.types[i]
#                        wasbits = false
#                        if typeof(fldty) === DataType
#                            fldty = fldty::DataType
#                            if isbits(fldty)
#                                fldsz = fldty.size
#                                if fldsz == 1
#                                    readfield(UInt8, s, xptr + Base.field_offset(t, i))
#                                elseif fldsz == 2
#                                    readfield(UInt16, s, xptr + Base.field_offset(t, i))
#                                elseif fldsz == 4
#                                    readfield(UInt32, s, xptr + Base.field_offset(t, i))
#                                elseif fldsz == 8
#                                    readfield(UInt64, s, xptr + Base.field_offset(t, i))
#                                else
#                                    readfield(fldty, s, xptr + Base.field_offset(t, i))
#                                end
#                                #write(s.io, getfield(x, i))
#                                wasbits = true
#                            end
#                        end
#                        if !wasbits && isdefined(x, i)
#                            xf = getfield(x, i)
#                            id = get!(s.serialized, xf, 0)::Int
#                            if id == 0 || id > xid
#                                needsreloc = true
#                            else
#                                push!(s.relocations, (xid, i, id))
#                            end
#                        end
#                    end
#                    if needsreloc
#                        push!(relocations, x)
#                    end
#                end
#            end
#        end
#
#        if reloc[1] != 0
#            push!(s.relocations, reloc)
#        end
#    end
#
#    # flush all of the pending relocations
#    writetag(s.io, UNDEFREF_TAG)
#    write(s.io, Int(length(s.relocations)))
#    write(s.io, s.relocations)
#    write(s.io, length(relocarrays))
#    for i = 1:length(relocarrays)
#        write(s.io, relocarrayid[i])
#        write(s.io, relocarrays[i])
#    end
#    empty!(s.relocations)
#    nothing
#end
#
#function serialize_tuple(s::SerializationState, t::Tuple)
#    l = length(t)
#    if l == 0
#        writetag(s.io, EMPTYTUPLE_TAG)
#    elseif l <= 255
#        writetag(s.io, TUPLE_TAG)
#        write(s.io, UInt8(l))
#    else
#        writetag(s.io, LONGTUPLE_TAG)
#        write(s.io, Int32(l))
#    end
#    for i = 1:l
#        fastserialize_any(s, t[i])
#    end
#    nothing
#end
#
#function serialize_array(s::SerializationState, a::Array, relocations)
#    elty = eltype(a)
#    writetag(s.io, ARRAY_TAG)
#    if elty !== UInt8
#        fastserialize_any(s, elty)
#    end
#    if ndims(a) != 1
#        fastserialize_any(s, size(a))
#    else
#        fastserialize_any(s, length(a))
#    end
#    if isbits(elty)
#        serialize_array_bits(s.io, a)
#    else
#        isa(s.io, IOBuffer) && Base.ensureroom(s.io, length(a) * (5 * Core.sizeof(UInt) + Core.sizeof(eltype(a))))
#        hasdata = false
#        for i = 1:length(a)
#            if isdefined(a, i)
#                hasdata = true
#                break
#            end
#        end
#        if hasdata
#            push!(relocations, a)
#        end
#    end
#    nothing
#end
#
#function serialize_expr(s::SerializationState, ex::Expr)
#    l = length(ex.args)
#    if l <= 255
#        writetag(s.io, EXPR_TAG)
#        write(s.io, UInt8(l))
#    else
#        writetag(s.io, LONGEXPR_TAG)
#        write(s.io, Int32(l))
#    end
#    fastserialize_any(s, ex.head)
#    fastserialize_any(s, ex.typ)
#    fastserialize_any(s, ex.args)
#    nothing
#end
#

############

type FastDeSerializationState{I<:IO}
    io::I
    deserialized::Vector{Any}
    FastDeSerializationState(io::I) = new(io, Vector{Any}())
end
FastDeSerializationState(io::IO) = FastDeSerializationState{typeof(io)}(io)

deserialize(s::IO) = fastdeserialize_any(FastDeSerializationState(s))

function fastdeserialize_any(s::FastDeSerializationState)
    pos = -1
    while true
        b = Int32(read(s.io, UInt8)::UInt8)
        #println(b, " => ", b == 0 ? "..." : desertag(b))
        if b == UNDEFREF_TAG
            if pos < 0
                pos = length(s.deserialized)
                continue
            else
                break
            end
        end

        if b == 0
            b = Int32(read(s.io, UInt8)::UInt8)
            x = desertag(b)
        elseif b >= VALUE_TAGS
            x = desertag(b)
        elseif b == DATATYPE_TAG
            x = deserialize_datatype(s)
        elseif b == SYMBOL_TAG
            x = symbol(read(s.io, UInt8, Int(read(s.io, UInt8)::UInt8)))
        elseif b == LONGSYMBOL_TAG
            x = symbol(read(s.io, UInt8, Int(read(s.io, Int32)::Int32)))
        elseif b == UNION_TAG
            x = deserialize_union(s)
        elseif b == MODULE_TAG
            x = deserialize_module(s)
        elseif b == ARRAY_TAG
            x = deserialize_array(s)
        elseif b == SIMPLEVECTOR_TAG
            l = read(s.io, Int)
            x = ccall(:jl_alloc_svec, Any, (Csize_t,), l)
        else
            if b == VALUE_TAG
                t = s.deserialized[read(s.io, Int)]::DataType
            else
                t = desertag(b)::DataType
            end
            nf = nfields(t)
            if nf == 0 && t.size > 0 # bitstype
                x = read(s.io, t)
            else
                x = ccall(:jl_new_struct_uninit, Any, (Any,), t)
                xptr = pointer_from_objref(x)
                for i = 1:nf
                    if !jl_field_isptr(t, i)
                        fldty = t.types[i]::DataType
                        @readfield(s, x, xptr, t, i, fldty)
                    end
                end
            end
        end
        #Core.Inference.println(x)
        push!(s.deserialized, x)
    end
    #Core.Inference.println(s.deserialized)
    for i = 1:read(s.io, Int)
        counter = read(s.io, Int) + pos
        field = read(s.io, Int)
        fid = read(s.io, Int)
        obj = s.deserialized[counter]
        fld = s.deserialized[fid > 0 ? fid : pos - fid]
        if isa(obj, SimpleVector)
            ptr = pointer_from_objref(obj) + field * sizeof(Ptr{Void})
            unsafe_store!(ptr, fld)
        else
            ccall(:jl_set_nth_field, Void, (Any, Csize_t, Any), obj, field-1, fld)
        end
    end
    for i = 1:read(s.io, Int)
        counter = read(s.io, Int) + pos
        a = s.deserialized[counter]::Array
        aptr = convert(Ptr{Any}, pointer(a))::Ptr{Any}
        nf = length(a)
        d = read!(s.io, Vector{Int}(nf))
        for j = 1:nf
            fid = d[j]
            if fid != 0
                unsafe_store!(aptr, s.deserialized[fid >= 0 ? fid : pos - fid], j)
            end
        end
    end
    #println(s.deserialized)
    retid = read(s.io, Int)
    return s.deserialized[retid >= 0 ? retid : pos - retid]
end


function deserialize_module(s::FastDeSerializationState)
    m = Main
    while true
        id_mname = read(s.io, Int)
        id_mname == 0 && return m
        mname = s.deserialized[id_mname]
        if !isdefined(m, mname)
            warn("Module $mname not defined on process $(myid())")  # an error seemingly fails
        end
        m = getfield(m, mname)::Module
    end
end

function deserialize_datatype(s::FastDeSerializationState)
    id_tname = read(s.io, Int)
    id_mod = read(s.io, Int)
    name = s.deserialized[id_tname]::Symbol
    mod = s.deserialized[id_mod]::Module
    ty = getfield(mod, name)
    if isempty(ty.parameters)
        return ty
    end
    count = read(s.io, Int)
    params = Vector{Any}(count)
    for i = 1:count
        params[i] = s.deserialized[read(s.io, Int)]
    end
    return ty{params...}
end

function deserialize_union(s::FastDeSerializationState)
    count = read(s.io, Int)
    types = Vector{Any}(count)
    for i = 1:count
        types[i] = s.deserialized[i]
    end
    return Union{types...}
end

function deserialize_array(s::FastDeSerializationState)
    elty = s.deserialized[read(s.io, Int)]
    nd = read(s.io, Int)
    if nd == 1
        d1 = read(s.io, Int)
        if elty !== Bool && isbits(elty)
            return read!(s.io, Array(elty, d1))::Array
        end
        dims = (Int(d1),)
    else
        dims = ntuple(i -> read(s.io, Int), nd)
    end
    if isbits(elty)
        # deserialize_array_bits
        n = prod(dims)::Int
        if elty === Bool && n>0
            A = Array(Bool, dims)
            i = 1
            while i <= n
                b = read(s.io, UInt8)::UInt8
                v = (b>>7) != 0
                count = b&0x7f
                nxt = i+count
                while i < nxt
                    A[i] = v; i+=1
                end
            end
        else
            A = read(s.io, elty, dims)
        end
    else
        A = Array(elty, dims)
    end
    return A
end

#
##indent_level = 0
#function fastdeserialize_any(s::FastDeSerializationState)
#    #global indent_level
#    b = Int32(read(s.io, UInt8)::UInt8)
#    if b == UNDEFREF_TAG
#        return s.deserialized[read(s.io, Int)]
#    end
#    first = length(s.deserialized) + 1
#    #indent_level += 1
#    while b != UNDEFREF_TAG
#        push!(s.deserialized, nothing)
#        idx = length(s.deserialized)
#        #println(" " ^ indent_level, idx, ' ', b, ' ', b != 0 ? desertag(b) : 0)
#        if b == 0
#            b = Int32(read(s.io, UInt8)::UInt8)
#            x = desertag(b)
#        elseif b >= VALUE_TAGS
#            x = desertag(b)
#        #elseif b == TUPLE_TAG
#        #    x = deserialize_tuple(s, Int(read(s.io, UInt8)::UInt8))
#        #elseif b == LONGTUPLE_TAG
#        #    x = deserialize_tuple(s, Int(read(s.io, Int32)::Int32))
#        elseif b == ARRAY_TAG
#            x = deserialize_array(s)
#        elseif b == DATATYPE_TAG
#            x = deserialize_datatype(s)
#        elseif b == SYMBOL_TAG
#            x = symbol(read(s.io, UInt8, Int(read(s.io, UInt8)::UInt8)))
#        elseif b == LONGSYMBOL_TAG
#            x = symbol(read(s.io, UInt8, Int(read(s.io, Int32)::Int32)))
#        elseif b == EXPR_TAG
#            x = deserialize_expr(s, Int(read(s.io, UInt8)::UInt8))
#        elseif b == LONGEXPR_TAG
#            x = deserialize_expr(s, Int(read(s.io, Int32)::Int32))
#        elseif b == SIMPLEVECTOR_TAG
#            x = deserialize_sv(s)
#        elseif b == UNION_TAG
#            x = deserialize_union(s)
#        elseif b == MODULE_TAG
#            x = deserialize_module(s)
#        else
#            if b == VALUE_TAG
#                t = fastdeserialize_any(s)::DataType
#            else
#                t = desertag(b)::DataType
#            end
#            nf = nfields(t)
#            if nf == 0 && t.size > 0 # bitstype
#                x = read(s.io, t)
#            else
#                x = ccall(:jl_new_struct_uninit, Any, (Any,), t)
#                xptr = pointer_from_objref(x)
#                for i = 1:nf
#                    fldty = t.types[i]
#                    if typeof(fldty) === DataType
#                        fldty = fldty::DataType
#                        if isbits(fldty::DataType)
#                            fldsz = fldty.size
#                            if fldsz == 1
#                                writefield(UInt8, s, xptr + Base.field_offset(t, i))
#                            elseif fldsz == 2
#                                writefield(UInt16, s, xptr + Base.field_offset(t, i))
#                            elseif fldsz == 4
#                                writefield(UInt32, s, xptr + Base.field_offset(t, i))
#                            elseif fldsz == 8
#                                writefield(UInt64, s, xptr + Base.field_offset(t, i))
#                            else
#                                writefield(fldty, s, xptr + Base.field_offset(t, i))
#                            end
#                            #ccall(:jl_set_nth_field, Void, (Any, Csize_t, Any), x, i-1, read(s.io, fldty))
#                        end
#                    end
#                end
#            end
#        end
#        s.deserialized[idx] = x
#        #println(idx, ' ', x)
#        b = Int32(read(s.io, UInt8)::UInt8)
#    end
#    #println(s.deserialized)
#    for i = 1:read(s.io, Int)
#        counter = read(s.io, Int)
#        field = read(s.io, Int)
#        id = read(s.io, Int)
#        ccall(:jl_set_nth_field, Void, (Any, Csize_t, Any), s.deserialized[counter], field-1, s.deserialized[id])
#    end
#    for i = 1:read(s.io, Int)
#        counter = read(s.io, Int)
#        a = s.deserialized[counter]::Array
#        nf = length(a)
#        d = read!(s.io, Vector{Int}(nf))
#        for j = 1:nf
#            id = d[j]
#            if id != 0
#                arrayset(a, s.deserialized[id], j)
#            end
#        end
#    end
#    #indent_level -= 1
#    return s.deserialized[first]
#end
#
#deserialize_tuple(s::SerializationState, len) = ntuple(i->deserialize(s), len)

#function writefield{T}(::Type{T}, s::FastDeSerializationState, fptr)
#    unsafe_store!(convert(Ptr{T}, fptr), read(s.io, T))
#    nothing
#end
#
#function deserialize_sv(s::FastDeSerializationState)
#    return svec((fastdeserialize_any(s)::Vector{Any})...)
#end
#
#function deserialize_array(s::FastDeSerializationState)
#    d1 = fastdeserialize_any(s)
#    if isa(d1, Type)
#        elty = d1
#        d1 = fastdeserialize_any(s)
#    else
#        elty = UInt8
#    end
#    if isa(d1, Integer)
#        if elty !== Bool && isbits(elty)
#            return read!(s.io, Array(elty, d1))::Array
#        end
#        dims = (Int(d1),)
#    else
#        dims = convert(Dims, d1)::Dims
#    end
#    if isbits(elty)
#        # deserialize_array_bits
#        n = prod(dims)::Int
#        if elty === Bool && n>0
#            A = Array(Bool, dims)
#            i = 1
#            while i <= n
#                b = read(s.io, UInt8)::UInt8
#                v = (b>>7) != 0
#                count = b&0x7f
#                nxt = i+count
#                while i < nxt
#                    A[i] = v; i+=1
#                end
#            end
#        else
#            A = read(s.io, elty, dims)
#        end
#    else
#        A = Array(elty, dims)
#    end
#    return A
#end
#
#
#function deserialize_expr(s::FastDeSerializationState, len)
#    hd = fastdeserialize_any(s)::Symbol
#    ty = fastdeserialize_any(s)
#    e = Expr(hd)
#    e.args = fastdeserialize_any(s)
#    e.typ = ty
#    e
#end
#


## MUCH OLDER CRAP ##########################

#serialize(s::SerializationState, p::Ptr) = serialize_any(s, oftype(p, C_NULL))

#function serialize{T,N,A<:Array}(s::SerializationState, a::SubArray{T,N,A})
#    if !isbits(T) || stride(a,1)!=1
#        return serialize(s, copy(a))
#    end
#    writetag(s.io, ARRAY_TAG)
#    serialize(s, T)
#    serialize(s, size(a))
#    serialize_array_data(s.io, a)
#end

#function serialize{T<:AbstractString}(s::SerializationState, ss::SubString{T})
#    # avoid saving a copy of the parent string, keeping the type of ss
#    serialize_any(s, convert(SubString{T}, convert(T,ss)))
#end

## Don't serialize the pointers
#function serialize(s::SerializationState, r::Regex)
#    serialize_type(s, typeof(r))
#    serialize(s, r.pattern)
#    serialize(s, r.compile_options)
#    serialize(s, r.match_options)
#end
#
#function serialize(s::SerializationState, n::BigInt)
#    serialize_type(s, BigInt)
#    serialize(s, base(62,n))
#end
#
#function serialize(s::SerializationState, n::BigFloat)
#    serialize_type(s, BigFloat)
#    serialize(s, string(n))
#end
#
#function serialize(s::SerializationState, t::Dict)
#    serialize_cycle(s, t) && return
#    serialize_type(s, typeof(t))
#    write(s.io, Int32(length(t)))
#    for (k,v) in t
#        serialize(s, k)
#        serialize(s, v)
#    end
#end
##
#function serialize(s::SerializationState, f::Function)
#    name = false
#    if isgeneric(f)
#        name = f.env.name
#    elseif isa(f.env,Symbol)
#        name = f.env
#    end
#    if isa(name,Symbol)
#        if isdefined(Base,name) && is(f,getfield(Base,name))
#            writetag(s.io, FUNCTION_TAG)
#            write(s.io, UInt8(0))
#            serialize(s, name)
#            return
#        end
#        mod = ()
#        if isa(f.env,Symbol)
#            mod = Core
#        elseif isdefined(f.env, :module) && isa(f.env.module, Module)
#            mod = f.env.module
#        elseif !is(f.env.defs, ())
#            mod = f.env.defs.func.code.module
#        end
#        if mod !== ()
#            if isdefined(mod,name) && is(f,getfield(mod,name))
#                # toplevel named func
#                writetag(s.io, FUNCTION_TAG)
#                write(s.io, UInt8(2))
#                serialize(s, mod)
#                serialize(s, name)
#                return
#            end
#        end
#        serialize_cycle(s, f) && return
#        writetag(s.io, FUNCTION_TAG)
#        write(s.io, UInt8(3))
#        serialize(s, f.env)
#    else
#        serialize_cycle(s, f) && return
#        writetag(s.io, FUNCTION_TAG)
#        write(s.io, UInt8(1))
#        linfo = f.code
#        @assert isa(linfo,LambdaStaticData)
#        serialize(s, linfo)
#        serialize(s, f.env)
#    end
#end
#
#const lambda_numbers = WeakKeyDict()
#lnumber_salt = 0
#function lambda_number(l::LambdaStaticData)
#    global lnumber_salt, lambda_numbers
#    if haskey(lambda_numbers, l)
#        return lambda_numbers[l]
#    end
#    # a hash function that always gives the same number to the same
#    # object on the same machine, and is unique over all machines.
#    ln = lnumber_salt+(UInt64(myid())<<44)
#    lnumber_salt += 1
#    lambda_numbers[l] = ln
#    return ln
#end
#
#function serialize(s::SerializationState, linfo::LambdaStaticData)
#    serialize_cycle(s, linfo) && return
#    writetag(s.io, LAMBDASTATICDATA_TAG)
#    serialize(s, lambda_number(linfo))
#    serialize(s, uncompressed_ast(linfo))
#    if isdefined(linfo.def, :roots)
#        serialize(s, linfo.def.roots::Vector{Any})
#    else
#        serialize(s, Any[])
#    end
#    serialize(s, linfo.sparams)
#    serialize(s, linfo.inferred)
#    serialize(s, linfo.module)
#    if isdefined(linfo, :capt)
#        serialize(s, linfo.capt)
#    else
#        serialize(s, nothing)
#    end
#end
#
#function serialize(s::SerializationState, t::Task)
#    serialize_cycle(s, t) && return
#    if istaskstarted(t) && !istaskdone(t)
#        error("cannot serialize a running Task")
#    end
#    state = [t.code,
#        t.storage,
#        t.state == :queued || t.state == :runnable ? (:runnable) : t.state,
#        t.result,
#        t.exception]
#    writetag(s.io, TASK_TAG)
#    for fld in state
#        serialize(s, fld)
#    end
#end

## deserializing values ##

#const known_lambda_data = Dict()
#
#function deserialize(s::SerializationState, ::Type{Function})
#    b = read(s.io, UInt8)::UInt8
#    if b==0
#        name = deserialize(s)::Symbol
#        if !isdefined(Base,name)
#            f = (args...)->error("function $name not defined on process $(myid())")
#        else
#            f = getfield(Base,name)::Function
#        end
#    elseif b==2
#        mod = deserialize(s)::Module
#        name = deserialize(s)::Symbol
#        if !isdefined(mod,name)
#            f = (args...)->error("function $name not defined on process $(myid())")
#        else
#            f = getfield(mod,name)::Function
#        end
#    elseif b==3
#        f = ccall(:jl_new_gf_internal, Any, (Any,), nothing)::Function
#        deserialize_cycle(s, f)
#        f.env = deserialize(s)
#    else
#        f = ccall(:jl_new_closure, Any, (Ptr{Void}, Ptr{Void}, Any),
#                  cglobal(:jl_trampoline), C_NULL, nothing)::Function
#        deserialize_cycle(s, f)
#        f.code = li = deserialize(s)
#        f.fptr = ccall(:jl_linfo_fptr, Ptr{Void}, (Any,), li)
#        f.env = deserialize(s)
#    end
#
#    return f
#end
#
#function deserialize(s::SerializationState, ::Type{LambdaStaticData})
#    lnumber = deserialize(s)
#    if haskey(known_lambda_data, lnumber)
#        linfo = known_lambda_data[lnumber]::LambdaStaticData
#        makenew = false
#    else
#        linfo = ccall(:jl_new_lambda_info, Any, (Ptr{Void}, Ptr{Void}, Ptr{Void}), C_NULL, C_NULL, C_NULL)::LambdaStaticData
#        makenew = true
#    end
#    deserialize_cycle(s, linfo)
#    ast = deserialize(s)::Expr
#    roots = deserialize(s)::Vector{Any}
#    sparams = deserialize(s)::SimpleVector
#    infr = deserialize(s)::Bool
#    mod = deserialize(s)::Module
#    capt = deserialize(s)
#    if makenew
#        linfo.ast = ast
#        linfo.sparams = sparams
#        linfo.inferred = infr
#        linfo.module = mod
#        linfo.roots = roots
#        if !is(capt,nothing)
#            linfo.capt = capt::Vector{Any}
#        end
#        known_lambda_data[lnumber] = linfo
#    end
#    return linfo
#end

#function deserialize(s::SerializationState, ::Type{Task})
#    t = Task(()->nothing)
#    deserialize_cycle(s, t)
#    t.code = deserialize(s)
#    t.storage = deserialize(s)
#    t.state = deserialize(s)
#    t.result = deserialize(s)
#    t.exception = deserialize(s)
#    t
#end

#function deserialize{K,V}(s::SerializationState, T::Type{Dict{K,V}})
#    n = read(s.io, Int32)
#    t = T(); sizehint!(t, n)
#    deserialize_cycle(s, t)
#    for i = 1:n
#        k = deserialize(s)
#        v = deserialize(s)
#        t[k] = v
#    end
#    return t
#end

#deserialize(s::SerializationState, ::Type{BigFloat}) = parse(BigFloat, deserialize(s))
#
#deserialize(s::SerializationState, ::Type{BigInt}) = get(GMP.tryparse_internal(BigInt, deserialize(s), 62, true))
#
#deserialize(s::SerializationState, ::Type{BigInt}) = get(GMP.tryparse_internal(BigInt, deserialize(s), 62, true))
#
#function deserialize(s::SerializationState, t::Type{Regex})
#    pattern = deserialize(s)
#    compile_options = deserialize(s)
#    match_options = deserialize(s)
#    Regex(pattern, compile_options, match_options)
#end

end
