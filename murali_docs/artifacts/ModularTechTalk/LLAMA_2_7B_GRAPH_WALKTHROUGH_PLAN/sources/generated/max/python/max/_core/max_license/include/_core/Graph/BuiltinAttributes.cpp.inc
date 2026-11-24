nb::class_<mlir::ArrayAttr, ::mlir::Attribute>(m, "ArrayAttr")
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, llvm::ArrayRef<Attribute>>(&mlir::ArrayAttr::get))), nb::arg("value"))
    .def_prop_ro("value", optional_pimpl(&mlir::ArrayAttr::getValue))
    .doc() = R"nbdoc(
        Syntax:
    
        ```
        array-attribute ::= `[` (attribute-value (`,` attribute-value)*)? `]`
        ```
    
        An array attribute is an attribute that represents a collection of attribute
        values.
    
        Examples:
    
        ```mlir
        []
        [10, i32]
        [affine_map<(d0, d1, d2) -> (d0, d1)>, i32, "string attribute"]
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<mlir::ArrayAttr>(), &typeid(mlir::ArrayAttr));
nb::class_<mlir::DictionaryAttr, ::mlir::Attribute>(m, "DictionaryAttr")
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, ArrayRef<NamedAttribute>>(&mlir::DictionaryAttr::get))), nb::arg("value") = ArrayRef<NamedAttribute>())
    .def_prop_ro("value", optional_pimpl(&mlir::DictionaryAttr::getValue))
    .doc() = R"nbdoc(
        Syntax:
    
        ```
        dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
        ```
    
        A dictionary attribute is an attribute that represents a sorted collection of
        named attribute values. The elements are sorted by name, and each name must be
        unique within the collection.
    
        Examples:
    
        ```mlir
        {}
        {attr_name = "string attribute"}
        {int_attr = 10, "string attr name" = "string attribute"}
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<mlir::DictionaryAttr>(), &typeid(mlir::DictionaryAttr));
nb::class_<mlir::FloatAttr, ::mlir::Attribute>(m, "FloatAttr")
    .def(nb::new_(&get_checked<mlir::FloatAttr, Type, const APFloat &>), nb::arg("type"), nb::arg("value"))
    .def(nb::new_(&get_checked<mlir::FloatAttr, Type, double>), nb::arg("type"), nb::arg("value"))
    .def_prop_ro("type", optional_pimpl(&mlir::FloatAttr::getType))
    .def_prop_ro("value", optional_pimpl(&mlir::FloatAttr::getValue))
    .doc() = R"nbdoc(
        Syntax:
    
        ```
        float-attribute ::= (float-literal (`:` float-type)?)
                          | (hexadecimal-literal `:` float-type)
        ```
    
        A float attribute is a literal attribute that represents a floating point
        value of the specified [float type](#floating-point-types). It can be
        represented in the hexadecimal form where the hexadecimal value is
        interpreted as bits of the underlying binary representation. This form is
        useful for representing infinity and NaN floating point values. To avoid
        confusion with integer attributes, hexadecimal literals _must_ be followed
        by a float type to define a float attribute.
    
        Examples:
    
        ```
        42.0         // float attribute defaults to f64 type
        42.0 : f32   // float attribute of f32 type
        0x7C00 : f16 // positive infinity
        0x7CFF : f16 // NaN (one of possible values)
        42 : f32     // Error: expected integer type
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<mlir::FloatAttr>(), &typeid(mlir::FloatAttr));
nb::class_<mlir::IntegerAttr, ::mlir::Attribute>(m, "IntegerAttr")
    .def(nb::new_(&get_checked<mlir::IntegerAttr, Type, const APInt &>), nb::arg("type"), nb::arg("value"))
    .def(nb::new_(infer_context(&get_checked<mlir::IntegerAttr, mlir::MLIRContext*, const APSInt &>)), nb::arg("value"))
    .def(nb::new_(&get_checked<mlir::IntegerAttr, Type, int64_t>), nb::arg("type"), nb::arg("value"))
    .def_prop_ro("type", optional_pimpl(&mlir::IntegerAttr::getType))
    .def_prop_ro("value", optional_pimpl(&mlir::IntegerAttr::getValue))
    .doc() = R"nbdoc(
        Syntax:
    
        ```
        integer-attribute ::= (integer-literal ( `:` (index-type | integer-type) )?)
                              | `true` | `false`
        ```
    
        An integer attribute is a literal attribute that represents an integral
        value of the specified integer or index type. `i1` integer attributes are
        treated as `boolean` attributes, and use a unique assembly format of either
        `true` or `false` depending on the value. The default type for non-boolean
        integer attributes, if a type is not specified, is signless 64-bit integer.
    
        Examples:
    
        ```mlir
        10 : i32
        10    // : i64 is implied here.
        true  // A bool, i.e. i1, value.
        false // A bool, i.e. i1, value.
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<mlir::IntegerAttr>(), &typeid(mlir::IntegerAttr));
nb::class_<mlir::StringAttr, ::mlir::Attribute>(m, "StringAttr")
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*>(&mlir::StringAttr::get))))
    .def(nb::new_(nb::overload_cast<const Twine &, Type>(&mlir::StringAttr::get)), nb::arg("bytes"), nb::arg("type"))
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, const Twine &>(&mlir::StringAttr::get))), nb::arg("bytes"))
    .def_prop_ro("value", optional_pimpl(&mlir::StringAttr::getValue))
    .def_prop_ro("type", optional_pimpl(&mlir::StringAttr::getType))
    .doc() = R"nbdoc(
        Syntax:
    
        ```
        string-attribute ::= string-literal (`:` type)?
        ```
    
        A string attribute is an attribute that represents a string literal value.
    
        Examples:
    
        ```mlir
        "An important string"
        "string with a type" : !dialect.string
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<mlir::StringAttr>(), &typeid(mlir::StringAttr));
    nb::implicitly_convertible<mlir::MLIRContext*, mlir::StringAttr>();
nb::class_<mlir::TypeAttr, ::mlir::Attribute>(m, "TypeAttr")
    .def(nb::new_(nb::overload_cast<Type>(&mlir::TypeAttr::get)), nb::arg("type"))
    .def_prop_ro("value", optional_pimpl(&mlir::TypeAttr::getValue))
    .doc() = R"nbdoc(
        Syntax:
    
        ```
        type-attribute ::= type
        ```
    
        A type attribute is an attribute that represents a
        [type object](#type-system).
    
        Examples:
    
        ```mlir
        i32
        !dialect.type
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<mlir::TypeAttr>(), &typeid(mlir::TypeAttr));
    nb::implicitly_convertible<Type, mlir::TypeAttr>();
nb::class_<mlir::UnitAttr, ::mlir::Attribute>(m, "UnitAttr")
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*>(&mlir::UnitAttr::get))))
    .doc() = R"nbdoc(
        Syntax:
    
        ```
        unit-attribute ::= `unit`
        ```
    
        A unit attribute is an attribute that represents a value of `unit` type. The
        `unit` type allows only one value forming a singleton set. This attribute
        value is used to represent attributes that only have meaning from their
        existence.
    
        One example of such an attribute could be the `swift.self` attribute. This
        attribute indicates that a function parameter is the self/context parameter.
        It could be represented as a [boolean attribute](#boolean-attribute)(true or
        false), but a value of false doesn't really bring any value. The parameter
        either is the self/context or it isn't.
    
    
        Examples:
    
        ```mlir
        // A unit attribute defined with the `unit` value specifier.
        func.func @verbose_form() attributes {dialectName.unitAttr = unit}
    
        // A unit attribute in an attribute dictionary can also be defined without
        // the value specifier.
        func.func @simple_form() attributes {dialectName.unitAttr}
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<mlir::UnitAttr>(), &typeid(mlir::UnitAttr));
    nb::implicitly_convertible<mlir::MLIRContext*, mlir::UnitAttr>();
