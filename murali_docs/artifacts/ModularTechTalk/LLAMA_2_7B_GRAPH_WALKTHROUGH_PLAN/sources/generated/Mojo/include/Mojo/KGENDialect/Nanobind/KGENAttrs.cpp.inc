nb::class_<M::KGEN::CastFromBuiltinAttr, ::mlir::Attribute>(m, "CastFromBuiltinAttr")
    .def(nb::new_(&get_checked<M::KGEN::CastFromBuiltinAttr, TypedAttr>), nb::arg("arg"))
    .def(nb::new_(&get_checked<M::KGEN::CastFromBuiltinAttr, TypedAttr, SIMDType>), nb::arg("arg"), nb::arg("type"))
    .def(nb::new_(infer_context(&get_checked<M::KGEN::CastFromBuiltinAttr, mlir::MLIRContext*, TypedAttr, SIMDType>)), nb::arg("arg"), nb::arg("type"))
    .def_prop_ro("arg", optional_pimpl(&M::KGEN::CastFromBuiltinAttr::getArg))
    .def_prop_ro("type", optional_pimpl(&M::KGEN::CastFromBuiltinAttr::getType))
    .doc() = R"nbdoc(
        The `#kgen.cast_from_builtin` attribute converts a builtin MLIR type to a
        POP type.
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::CastFromBuiltinAttr>(), &typeid(M::KGEN::CastFromBuiltinAttr));
    nb::implicitly_convertible<TypedAttr, M::KGEN::CastFromBuiltinAttr>();
nb::class_<M::KGEN::CastToBuiltinAttr, ::mlir::Attribute>(m, "CastToBuiltinAttr")
    .def(nb::new_(&get_checked<M::KGEN::CastToBuiltinAttr, TypedAttr>), nb::arg("arg"))
    .def(nb::new_(&get_checked<M::KGEN::CastToBuiltinAttr, TypedAttr, Type>), nb::arg("arg"), nb::arg("type"))
    .def(nb::new_(infer_context(&get_checked<M::KGEN::CastToBuiltinAttr, mlir::MLIRContext*, TypedAttr, Type>)), nb::arg("arg"), nb::arg("type"))
    .def_prop_ro("arg", optional_pimpl(&M::KGEN::CastToBuiltinAttr::getArg))
    .def_prop_ro("type", optional_pimpl(&M::KGEN::CastToBuiltinAttr::getType))
    .doc() = R"nbdoc(
        The `#kgen.cast_to_builtin` attribute converts a POP type to a builtin MLIR
        type.
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::CastToBuiltinAttr>(), &typeid(M::KGEN::CastToBuiltinAttr));
    nb::implicitly_convertible<TypedAttr, M::KGEN::CastToBuiltinAttr>();
nb::class_<M::KGEN::ConstraintAttr, ::mlir::Attribute>(m, "ConstraintAttr")
    .def(nb::new_(&get_checked<M::KGEN::ConstraintAttr, TypedAttr, LocationAttr, StringAttr>), nb::arg("proposition"), nb::arg("loc"), nb::arg("message"))
    .def(nb::new_(infer_context(&get_checked<M::KGEN::ConstraintAttr, mlir::MLIRContext*, TypedAttr, LocationAttr, StringAttr>)), nb::arg("proposition"), nb::arg("loc"), nb::arg("message"))
    .def_prop_ro("proposition", optional_pimpl(&M::KGEN::ConstraintAttr::getProposition))
    .def_prop_ro("loc", optional_pimpl(&M::KGEN::ConstraintAttr::getLoc))
    .def_prop_ro("message", optional_pimpl(&M::KGEN::ConstraintAttr::getMessage))
    .doc() = R"nbdoc(
        The `#kgen.constraint` attribute represents a proposition that should hold
        and a location for where this constraint was declared, which is useful for
        error reporting.
    
        The proposition is an i1-typed parameter expression.
    
        The optional message is a user-provided string (from
        `where (cond, "message")` syntax) that is surfaced in the diagnostic when
        the constraint fails. It is null for constraints without a user message
        (including all compiler-synthesized constraints).
    
        Example:
    
        ```mlir
        #kgen.constraint<1, loc("file.mojo":10:5)>
        #kgen.constraint<1, loc("file.mojo":10:5), "must be positive">
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::ConstraintAttr>(), &typeid(M::KGEN::ConstraintAttr));
nb::class_<M::KGEN::ParamDeclArrayAttr, ::mlir::Attribute>(m, "ParamDeclArrayAttr")
    .def(nb::new_(nb::overload_cast<ParamDeclAttr>(&M::KGEN::ParamDeclArrayAttr::get)), nb::arg("param_decl"))
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, llvm::ArrayRef<ParamDeclAttr>>(&M::KGEN::ParamDeclArrayAttr::get))), nb::arg("value"))
    .def_prop_ro("value", optional_pimpl(&M::KGEN::ParamDeclArrayAttr::getValue))
    .doc() = R"nbdoc()nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::ParamDeclArrayAttr>(), &typeid(M::KGEN::ParamDeclArrayAttr));
    nb::implicitly_convertible<ParamDeclAttr, M::KGEN::ParamDeclArrayAttr>();
nb::class_<M::KGEN::ParamDeclAttr, ::mlir::Attribute>(m, "ParamDeclAttr")
    .def(nb::new_(nb::overload_cast<ParamDeclRefAttr>(&M::KGEN::ParamDeclAttr::get)), nb::arg("ref"))
    .def(nb::new_(nb::overload_cast<StringAttr, Type>(&M::KGEN::ParamDeclAttr::get)), nb::arg("name"), nb::arg("type"))
    .def(nb::new_(nb::overload_cast<StringRef, Type>(&M::KGEN::ParamDeclAttr::get)), nb::arg("name"), nb::arg("type"))
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, StringAttr, Type>(&M::KGEN::ParamDeclAttr::get))), nb::arg("name"), nb::arg("type"))
    .def_prop_ro("name", optional_pimpl(&M::KGEN::ParamDeclAttr::getName))
    .def_prop_ro("type", optional_pimpl(&M::KGEN::ParamDeclAttr::getType))
    .doc() = R"nbdoc(
        This is a declaration of a parameter in the meta-programming domain for
        generators and related infrastructure.  These are typically owned by
        kgen.generator instances or other things that produce new attributes.
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::ParamDeclAttr>(), &typeid(M::KGEN::ParamDeclAttr));
    nb::implicitly_convertible<ParamDeclRefAttr, M::KGEN::ParamDeclAttr>();
nb::class_<M::KGEN::ParamDeclRefAttr, ::mlir::Attribute>(m, "ParamDeclRefAttr")
    .def(nb::new_(nb::overload_cast<ParamDeclAttr>(&M::KGEN::ParamDeclRefAttr::get)), nb::arg("decl"))
    .def(nb::new_(nb::overload_cast<StringAttr, Type>(&M::KGEN::ParamDeclRefAttr::get)), nb::arg("name"), nb::arg("type"))
    .def(nb::new_(nb::overload_cast<StringRef, Type>(&M::KGEN::ParamDeclRefAttr::get)), nb::arg("name"), nb::arg("type"))
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, StringAttr, mlir::Type>(&M::KGEN::ParamDeclRefAttr::get))), nb::arg("name"), nb::arg("type"))
    .def_prop_ro("name", optional_pimpl(&M::KGEN::ParamDeclRefAttr::getName))
    .def_prop_ro("type", optional_pimpl(&M::KGEN::ParamDeclRefAttr::getType))
    .doc() = R"nbdoc(
        The `#kgen.param.decl.ref` attribute is a typed attribute that represents
        a reference to a declared parameter. It contains the type of the referenced
        parameter and its name.
    
        Example:
    
        ```mlir
        // A reference to parameter "p" with type "i1".
        #kgen.param.decl.ref<"p"> : i1
        ```
    
        There are special rules governing when these can appear, or when
        ParamIndexRefAttr must be used instead, see DCRTODS.
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::ParamDeclRefAttr>(), &typeid(M::KGEN::ParamDeclRefAttr));
    nb::implicitly_convertible<ParamDeclAttr, M::KGEN::ParamDeclRefAttr>();
nb::class_<M::KGEN::ParamIndexRefAttr, ::mlir::Attribute>(m, "ParamIndexRefAttr")
    .def(nb::new_(nb::overload_cast<size_t, Type>(&M::KGEN::ParamIndexRefAttr::get)), nb::arg("index"), nb::arg("type"))
    .def(nb::new_(nb::overload_cast<size_t, size_t, Type>(&M::KGEN::ParamIndexRefAttr::get)), nb::arg("depth"), nb::arg("index"), nb::arg("type"))
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, size_t, size_t, mlir::Type>(&M::KGEN::ParamIndexRefAttr::get))), nb::arg("depth"), nb::arg("index"), nb::arg("type"))
    .def_prop_ro("depth", optional_pimpl(&M::KGEN::ParamIndexRefAttr::getDepth))
    .def_prop_ro("index", optional_pimpl(&M::KGEN::ParamIndexRefAttr::getIndex))
    .def_prop_ro("type", optional_pimpl(&M::KGEN::ParamIndexRefAttr::getType))
    .doc() = R"nbdoc(
        The `#kgen.param.index.ref` attribute is a reference to an input
        parameter of an enclosing signature. This attribute can only be used inside
        a `GeneratorType`. This attribute contains two fields:
    
        - depth: Which containing signature contains the parameter we're referring
          to. Non-negative integer. Zero means the nearest containing signature, one
          means the signature containing that one, etc.
          Note they cannot refer to any op's parameter-decls, and you cannot always
          use a depth to refer to surrounding scopes, see DCRTODS.
        - index: index of the parameter decl in that signature (non-negative
          integer).
    
        Example:
    
        ```mlir
        // Second input parameter of the nearest signature.
        #kgen.param.index.ref<0, 1> : index
        // Second input parameter of next enclosing signature.
        #kgen.param.index.ref<1, 1> : !lit.struct<@Int>
        ```
    
        The latter would appear in something like this:
    
        ```
        comptime bar: def[
          D: DType,
          N: Int,
          f: def[Y: AnyType](Y, SIMD[D, N])->None
        ](...) = ...
        ```
    
        The `SIMD[D, N]`'s `N` is a #kgen.param.index.ref<1, 1> : !lit.struct<@Int>,
        because `N` is declared at index 1 of the signature one level out. `D` is
        that signature's index 0, so it is a #kgen.param.index.ref<1, 0>.
    
        But, per DCRTODS, it can NOT be used inside a generator's body to refer to
        one of the generator's parameters, like this:
    
        ```
        def foo[X: AnyType](x: X):
            comptime zork: def[...(
              # Cannot have: #kgen.param.index.ref<1, 0> : !lit.struct<@Int>
            )->None = ...
        ```
    
        These depths must be carefully handled and adjusted when dealing with
        multiple signatures or scopes, see STCHDDDOS.
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::ParamIndexRefAttr>(), &typeid(M::KGEN::ParamIndexRefAttr));
nb::class_<M::KGEN::ParamOperatorAttr, ::mlir::Attribute>(m, "ParamOperatorAttr")
    .def(nb::new_(&get_checked<M::KGEN::ParamOperatorAttr, POC, ArrayRef<TypedAttr>>), nb::arg("opcode"), nb::arg("operands"))
    .def(nb::new_(&get_checked<M::KGEN::ParamOperatorAttr, POC, ArrayRef<TypedAttr>, Type>), nb::arg("opcode"), nb::arg("operands"), nb::arg("type"))
    .def(nb::new_(&get_checked<M::KGEN::ParamOperatorAttr, POC, TypedAttr, TypedAttr>), nb::arg("opcode"), nb::arg("lhs"), nb::arg("rhs"))
    .def(nb::new_(infer_context(&get_checked<M::KGEN::ParamOperatorAttr, mlir::MLIRContext*, POC, ArrayRef<TypedAttr>, Type>)), nb::arg("opcode"), nb::arg("operands"), nb::arg("type"))
    .def_prop_ro("opcode", optional_pimpl(&M::KGEN::ParamOperatorAttr::getOpcode))
    .def_prop_ro("operands", optional_pimpl(&M::KGEN::ParamOperatorAttr::getOperands))
    .def_prop_ro("type", optional_pimpl(&M::KGEN::ParamOperatorAttr::getType))
    .doc() = R"nbdoc()nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::ParamOperatorAttr>(), &typeid(M::KGEN::ParamOperatorAttr));
nb::class_<M::KGEN::ParameterExprArrayAttr, ::mlir::Attribute>(m, "ParameterExprArrayAttr")
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, llvm::ArrayRef<TypedAttr>>(&M::KGEN::ParameterExprArrayAttr::get))), nb::arg("value"))
    .def_prop_ro("value", optional_pimpl(&M::KGEN::ParameterExprArrayAttr::getValue))
    .doc() = R"nbdoc()nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::ParameterExprArrayAttr>(), &typeid(M::KGEN::ParameterExprArrayAttr));
nb::class_<M::KGEN::PogListAttr, ::mlir::Attribute>(m, "PogListAttr")
    .def(nb::new_(infer_context(&get_checked<M::KGEN::PogListAttr, mlir::MLIRContext*>)))
    .def(nb::new_(infer_context(&get_checked<M::KGEN::PogListAttr, mlir::MLIRContext*, size_t>)), nb::arg("num_pogs"))
    .def(nb::new_(infer_context(&get_checked<M::KGEN::PogListAttr, mlir::MLIRContext*, ArrayRef<PogMetadataAttr>>)), nb::arg("pogs"))
    .def(nb::new_(infer_context(&get_checked<M::KGEN::PogListAttr, mlir::MLIRContext*, size_t, ArrayRef<ConstraintAttr>>)), nb::arg("num_pogs"), nb::arg("body_constraints"))
    .def(nb::new_(infer_context(&get_checked<M::KGEN::PogListAttr, mlir::MLIRContext*, ArrayRef<StringAttr>, ArrayRef<PassingKind>>)), nb::arg("names"), nb::arg("passing_kinds"))
    .def(nb::new_(infer_context(&get_checked<M::KGEN::PogListAttr, mlir::MLIRContext*, llvm::ArrayRef<PogMetadataAttr>, llvm::ArrayRef<ConstraintAttr>, ArgConvention>)), nb::arg("pogs"), nb::arg("body_constraints"), nb::arg("orig_variadic_convention"))
    .def(nb::new_(infer_context(&get_checked<M::KGEN::PogListAttr, mlir::MLIRContext*, ArrayRef<StringAttr>, ArrayRef<PassingKind>, ArrayRef<VariadicKind>, ArrayRef<TypedAttr>, std::optional<ArgConvention>, ArrayRef<ConstraintAttr>>)), nb::arg("names"), nb::arg("passing_kinds"), nb::arg("variadics"), nb::arg("defaults"), nb::arg("orig_variadic_convention"), nb::arg("body_constraints"))
    .def_prop_ro("pogs", optional_pimpl(&M::KGEN::PogListAttr::getPogs))
    .def_prop_ro("body_constraints", optional_pimpl(&M::KGEN::PogListAttr::getBodyConstraints))
    .def_prop_ro("orig_variadic_convention", optional_pimpl(&M::KGEN::PogListAttr::getOrigVariadicConvention))
    .doc() = R"nbdoc(
        The `#kgen.pog_list` attribute contains metadata about an argument or
        parameter list of a function, including names, passing kinds, default
        values, and information about variadic arguments.
        The positional default values correspond to the trailing positional
        (i.e. pos-only or pos-or-kw) args/params, and the keyword-only default
        values similarly correspond to the trailing keyword-only args/params.
    
        This attribute is the `metadata` field of `GeneratorType`.
    
        Example:
    
        ```mlir
        #kgen.pog_list<
          ["a", "b", "c", "d"],
          [pos, pos_or_kw, kw, kw],
        >
        ```
    
        The `origVariadicConvention` indicates whether the original argument
        convention of a VariadicList or VariadicPack, e.g. "mut *args: Int".
    
        Optional `bodyConstraints` holds constraints that are enforced by the body
        of the generator type that carries this list as metadata.
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::PogListAttr>(), &typeid(M::KGEN::PogListAttr));
    nb::implicitly_convertible<mlir::MLIRContext*, M::KGEN::PogListAttr>();
nb::class_<M::KGEN::PogMetadataAttr, ::mlir::Attribute>(m, "PogMetadataAttr")
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*>(&M::KGEN::PogMetadataAttr::get))))
    .def(nb::new_(nb::overload_cast<StringAttr, PassingKind, VariadicKind, TypedAttr>(&M::KGEN::PogMetadataAttr::get)), nb::arg("name"), nb::arg("passing_kind"), nb::arg("variadic") = VariadicKind(VariadicKind::None), nb::arg("default_value") = TypedAttr())
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, StringAttr, PassingKind, VariadicKind, TypedAttr>(&M::KGEN::PogMetadataAttr::get))), nb::arg("name"), nb::arg("passing_kind"), nb::arg("variadic"), nb::arg("default_value"))
    .def_prop_ro("name", optional_pimpl(&M::KGEN::PogMetadataAttr::getName))
    .def_prop_ro("passing_kind", optional_pimpl(&M::KGEN::PogMetadataAttr::getPassingKind))
    .def_prop_ro("variadic", optional_pimpl(&M::KGEN::PogMetadataAttr::getVariadic))
    .def_prop_ro("default_value", optional_pimpl(&M::KGEN::PogMetadataAttr::getDefaultValue))
    .doc() = R"nbdoc(
        The `#kgen.pog_metadata` attribute contains metadata about an argument or
        parameter of a function, including the name, passing kind, variadicness, and
        the default value (if present).
    
        Example:
    
        ```mlir
        #kgen.pog_metadata<"some_keyword_param", pos_or_kw, false, 42>
        #kgen.pog_metadata<"some_variadic_param", pos_or_kw, true>
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::PogMetadataAttr>(), &typeid(M::KGEN::PogMetadataAttr));
    nb::implicitly_convertible<mlir::MLIRContext*, M::KGEN::PogMetadataAttr>();
nb::class_<M::KGEN::SIMDAttr, ::mlir::Attribute>(m, "SIMDAttr")
    .def(nb::new_(&get_checked<M::KGEN::SIMDAttr, ArrayRef<DTypeValue>, SIMDType>), nb::arg("values"), nb::arg("type"))
    .def(nb::new_(&get_checked<M::KGEN::SIMDAttr, DTypeValue, SIMDType>), nb::arg("value"), nb::arg("type"))
    .def(nb::new_(&get_checked<M::KGEN::SIMDAttr, uint64_t, SIMDType>), nb::arg("int_val"), nb::arg("type"))
    .def(nb::new_(infer_context(&get_checked<M::KGEN::SIMDAttr, mlir::MLIRContext*, llvm::ArrayRef<DTypeValue>, SIMDType>)), nb::arg("values"), nb::arg("type"))
    .def_prop_ro("values", optional_pimpl(&M::KGEN::SIMDAttr::getValues))
    .def_prop_ro("type", optional_pimpl(&M::KGEN::SIMDAttr::getType))
    .doc() = R"nbdoc(
        The `#kgen.simd` attribute represents a constant SIMD vector value. It
        contains `N` values of a particular dtype. Only integer, floating point, and
        bool dtypes are supported.
    
        Example:
    
        ```mlir
        #kgen.simd<1, 2> : !kgen.simd<2, si32>
        #kgen.simd<1.5, 2.5> : !kgen.simd<2, f64>
        #kgen.simd<true, false> : !kgen.simd<2, bool>
        ```
    
        When all values of the SIMD vector are equal, the attribute has a special
        splat syntax:
    
        ```mlir
        #kgen<simd 0> : !kgen.simd<4, si32>
        #kgen<simd "1.5"> : !kgen.simd<4, f32>
        #kgen<simd false> : !kgen.simd<4, bool>
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::KGEN::SIMDAttr>(), &typeid(M::KGEN::SIMDAttr));
