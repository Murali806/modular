nb::class_<M::MOSH::ParamFromValueOp, ::mlir::OpState>(m, "ParamFromValueOp")
    .def(nb::new_(&create_op<M::MOSH::ParamFromValueOp, mlir::Location, M::KGEN::ParamDeclAttr, mlir::TypedValue<mlir::Type>>), nb::arg("builder"), nb::arg("location"), nb::arg("param_decl"), nb::arg("value"))
    .def_prop_rw("param_decl", optional_pimpl(&M::MOSH::ParamFromValueOp::getParamDecl), &M::MOSH::ParamFromValueOp::setParamDeclAttr)
    .def_prop_ro("value", optional_pimpl(&M::MOSH::ParamFromValueOp::getValue))
    .doc() = R"nbdoc()nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::MOSH::ParamFromValueOp>(), &typeid(M::MOSH::ParamFromValueOp));
nb::class_<M::MOSH::ParamToValueOp, ::mlir::OpState>(m, "ParamToValueOp")
    .def(nb::new_(&create_op<M::MOSH::ParamToValueOp, mlir::Location, mlir::Type, TypedAttr>), nb::arg("builder"), nb::arg("location"), nb::arg("result"), nb::arg("value"))
    .def_prop_rw("value", optional_pimpl(&M::MOSH::ParamToValueOp::getValue), &M::MOSH::ParamToValueOp::setValueAttr)
    .doc() = R"nbdoc(
        The `mo.param.to_value` operation materializes the value of a parameter
        expression as an SSA value.
    
        Example:
    
        ```mlir
        %idx   = mo.param.to_value = <D0>
        %shape = mo.param.to_value: !mosh.ape = <[10, D1, 20]>
        ```
    
        The output type annotation may be omitted if it's `index`.
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::MOSH::ParamToValueOp>(), &typeid(M::MOSH::ParamToValueOp));
nb::class_<M::MOSH::ShapeAttr, ::mlir::Attribute>(m, "ShapeAttr")
    .def(nb::new_(&get_checked<M::MOSH::ShapeAttr, ArrayRef<TypedAttr>, MOSH::ShapeType>), nb::arg("values"), nb::arg("type"))
    .def(nb::new_(&get_checked<M::MOSH::ShapeAttr, ArrayRef<int64_t>, MOSH::ShapeType>), nb::arg("values"), nb::arg("type"))
    .def(nb::new_(&get_checked<M::MOSH::ShapeAttr, IntArrayElementsAttr, MOSH::ShapeType>), nb::arg("int_dims"), nb::arg("type"))
    .def(nb::new_(infer_context(&get_checked<M::MOSH::ShapeAttr, mlir::MLIRContext*, llvm::ArrayRef<TypedAttr>, MOSH::ShapeType>)), nb::arg("values"), nb::arg("type"))
    .def_prop_ro("values", optional_pimpl(&M::MOSH::ShapeAttr::getValues))
    .def_prop_ro("type", optional_pimpl(&M::MOSH::ShapeAttr::getType))
    .doc() = R"nbdoc(
        The `#mosh.ape` attribute contains a variable number of `TypedAttr`s, each
        of which has a kgen.scalar<si64> type. The type of this attribute is
        always `!mosh.ape`.
    
        Each dimension `TypedAttr` inside `values` is one of:
        1. `KGEN::ParamDeclRefAttr` for a dimension parameter, e.g., `D0`.
        2. `KGEN::ParamOperatorAttr` for a dimension parameter expression, e.g.,
        `add(D1, 2)`.
        3. `KGEN::SIMDAttr` for a concrete integer dimension, e.g., `42` (typed as
        `kgen.scalar<si64>`).
    
        -1 denotes a dimension inferred from the others and the element count. At
        most one dimension can be -1.
    
        Example:
    
        ```mlir
        kgen.param.declare N = <3>
        #mosh<ape[1, ?, N]> : !mosh.ape
        #mosh<ape[3, -1, 42]> : !mosh.ape
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::MOSH::ShapeAttr>(), &typeid(M::MOSH::ShapeAttr));
nb::class_<M::MOSH::ShapeType, ::mlir::Type>(m, "ShapeType")
    .def(nb::new_([]() {
                    return mlir::OpBuilder(current_context()).getType<::M::MOSH::ShapeType>();
                }))
    .doc() = R"nbdoc()nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::MOSH::ShapeType>(), &typeid(M::MOSH::ShapeType));
