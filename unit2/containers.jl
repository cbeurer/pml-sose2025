struct Container{T}
    value::T
    values::AbstractArray{T}

    Container{T}(value::T) where {T <: Number} = new(value, Vector{T}(undef, 0))
end

Container(value::T) where {T <: Number} = Container{T}(value)

function addvalue(c::Container{T}, value::T) where {T <: Number}
    push!(c.values, value)
end

c = Container(0.5)
println("c: \n", c, "\n")

addvalue(c, 0.1)
println("c after addvalue(c, 0.1): \n", c, "\n")

addvalue(c, 0.5)
println("c after the second addvalue(c, 0.1): \n", c, "\n")