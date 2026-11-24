nb::class_<M::MO::BufferType, ::mlir::Type>(m, "BufferType")
    .def(nb::new_(&get_checked<M::MO::BufferType, TensorType>), nb::arg("tensor_type"))
    .def(nb::new_(&get_checked<M::MO::BufferType, MOSH::ShapeAttr, DType, DeviceRefAttr, DictionaryAttr>), nb::arg("shape"), nb::arg("dtype"), nb::arg("device_ref") = DeviceRefAttr(), nb::arg("metadata") = DictionaryAttr())
    .def(nb::new_(&get_checked<M::MO::BufferType, MOSH::ShapeAttr, Type, DeviceRefAttr, DictionaryAttr>), nb::arg("shape"), nb::arg("element_type"), nb::arg("device_ref") = DeviceRefAttr(), nb::arg("metadata") = DictionaryAttr())
    .def(nb::new_(&get_checked<M::MO::BufferType, ArrayRef<TypedAttr>, Type, DeviceRefAttr, DictionaryAttr>), nb::arg("shape"), nb::arg("element_type"), nb::arg("device_ref") = DeviceRefAttr(), nb::arg("metadata") = DictionaryAttr())
    .def(nb::new_(infer_context(&get_checked<M::MO::BufferType, mlir::MLIRContext*, MOSH::ShapeAttr, DType, DeviceRefAttr, DictionaryAttr>)), nb::arg("shape"), nb::arg("dtype"), nb::arg("device_ref"), nb::arg("metadata"))
    .def_prop_ro("shape", optional_pimpl(&M::MO::BufferType::getShape))
    .def_prop_ro("dtype", optional_pimpl(&M::MO::BufferType::getDtype))
    .def_prop_ro("device_ref", optional_pimpl(&M::MO::BufferType::getDeviceRef))
    .def_prop_ro("metadata", optional_pimpl(&M::MO::BufferType::getMetadata))
    .doc() = R"nbdoc(
        A mutable analogue of `!mo.tensor`; with mo.mutable.load and
        mo.mutable.store it models in-place operations. `shape` is a statically
        ranked `MOSH::ShapeAttr`; dtype `invalid` means unknown.
    
        Examples:
        ```mlir
        !mo.buffer<[4, 16], f32>    // static shape
        !mo.buffer<[N, N, 6], i32>  // parameterized shape
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::MO::BufferType>(), &typeid(M::MO::BufferType));
    nb::implicitly_convertible<TensorType, M::MO::BufferType>();
nb::class_<M::MO::BundleType, ::mlir::Type>(m, "BundleType")
    .def(nb::new_(infer_context(&get_checked<M::MO::BundleType, mlir::MLIRContext*, llvm::ArrayRef<Type>>)), nb::arg("element_types"))
    .def_prop_ro("element_types", optional_pimpl(&M::MO::BundleType::getElementTypes))
    .doc() = R"nbdoc(
        Bundles per-device `!mo.tensor`s into one SSA value. Elements may differ in
        device, shape, or dtype.
    
        The verbose syntax lists each element and is used for non-uniform
        elements:
    
        ```mlir
        !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>]>
        ```
    
        The compact syntax is used when all elements share shape, dtype, device
        label, and metadata, with contiguous device IDs (`ids[i] == ids[0] + i`):
    
        ```mlir
        !mo.bundle<[3], f32, gpu:0-1>
        ```
    
        A single element omits the range, and on the default host device omits
        the device too:
    
        ```mlir
        !mo.bundle<[3], f32, gpu:0>   // single element, explicit device
        !mo.bundle<[3], f32>          // single element, default host device
        ```
    
        Both forms store the same `elementTypes`.
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::MO::BundleType>(), &typeid(M::MO::BundleType));
nb::class_<M::MO::ChainType, ::mlir::Type>(m, "ChainType")
    .def(nb::new_([]() {
                    return mlir::OpBuilder(current_context()).getType<::M::MO::ChainType>();
                }))
    .doc() = R"nbdoc(
        Every side-effecting MO op consumes and produces a chain.
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::MO::ChainType>(), &typeid(M::MO::ChainType));
nb::class_<M::MO::OpaqueType, ::mlir::Type>(m, "OpaqueType")
    .def(nb::new_(nb::overload_cast<StringAttr>(&M::MO::OpaqueType::get)), nb::arg("symbol"))
    .def(nb::new_(nb::overload_cast<StringAttr, DictionaryAttr>(&M::MO::OpaqueType::get)), nb::arg("symbol"), nb::arg("parameters"))
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, StringAttr, DictionaryAttr>(&M::MO::OpaqueType::get))), nb::arg("symbol"), nb::arg("parameters"))
    .def_prop_ro("symbol", optional_pimpl(&M::MO::OpaqueType::getSymbol))
    .def_prop_ro("parameters", optional_pimpl(&M::MO::OpaqueType::getParameters))
    .doc() = R"nbdoc(
        A custom user-defined type.
          Example:
          ```mlir
            !mo.opaque<"my_list">
            !mo.opaque<"my_list", {foo = 42}>
          ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::MO::OpaqueType>(), &typeid(M::MO::OpaqueType));
    nb::implicitly_convertible<StringAttr, M::MO::OpaqueType>();
nb::class_<M::MO::ScalarType, ::mlir::Type>(m, "ScalarType")
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, DType>(&M::MO::ScalarType::get))), nb::arg("dtype"))
    .def_prop_ro("dtype", optional_pimpl(&M::MO::ScalarType::getDtype))
    .doc() = R"nbdoc()nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::MO::ScalarType>(), &typeid(M::MO::ScalarType));
nb::class_<M::MO::TensorType, ::mlir::Type>(m, "TensorType")
    .def(nb::new_(&get_checked<M::MO::TensorType, MOSH::ShapeAttr, DType, DeviceRefAttr, DictionaryAttr>), nb::arg("shape"), nb::arg("dtype"), nb::arg("device_ref") = DeviceRefAttr(), nb::arg("metadata") = DictionaryAttr())
    .def(nb::new_(&get_checked<M::MO::TensorType, MOSH::ShapeAttr, Type, DeviceRefAttr, DictionaryAttr>), nb::arg("shape"), nb::arg("element_type"), nb::arg("device_ref") = DeviceRefAttr(), nb::arg("metadata") = DictionaryAttr())
    .def(nb::new_(&get_checked<M::MO::TensorType, ArrayRef<int64_t>, Type, DeviceRefAttr, DictionaryAttr>), nb::arg("shape"), nb::arg("element_type"), nb::arg("device_ref") = DeviceRefAttr(), nb::arg("metadata") = DictionaryAttr())
    .def(nb::new_(&get_checked<M::MO::TensorType, ArrayRef<TypedAttr>, Type, DeviceRefAttr, DictionaryAttr>), nb::arg("shape"), nb::arg("element_type"), nb::arg("device_ref") = DeviceRefAttr(), nb::arg("metadata") = DictionaryAttr())
    .def(nb::new_(infer_context(&get_checked<M::MO::TensorType, mlir::MLIRContext*, MOSH::ShapeAttr, DType, DeviceRefAttr, DictionaryAttr>)), nb::arg("shape"), nb::arg("dtype"), nb::arg("device_ref"), nb::arg("metadata"))
    .def(nb::new_(infer_context(&get_checked<M::MO::TensorType, mlir::MLIRContext*, ArrayRef<int64_t>, DType, DeviceRefAttr, DictionaryAttr>)), nb::arg("shape"), nb::arg("dtype"), nb::arg("device_ref") = DeviceRefAttr(), nb::arg("metadata") = DictionaryAttr())
    .def(nb::new_(infer_context(&get_checked<M::MO::TensorType, mlir::MLIRContext*, ArrayRef<TypedAttr>, DType, DeviceRefAttr, DictionaryAttr>)), nb::arg("shape"), nb::arg("dtype"), nb::arg("device_ref") = DeviceRefAttr(), nb::arg("metadata") = DictionaryAttr())
    .def_prop_ro("shape", optional_pimpl(&M::MO::TensorType::getShape))
    .def_prop_ro("dtype", optional_pimpl(&M::MO::TensorType::getDtype))
    .def_prop_ro("device_ref", optional_pimpl(&M::MO::TensorType::getDeviceRef))
    .def_prop_ro("metadata", optional_pimpl(&M::MO::TensorType::getMetadata))
    .doc() = R"nbdoc(
        A tensor's shape and element type, an optional device ref, and an optional
        metadata dictionary (e.g. layout).
    
        `shape` is a `MOSH::ShapeAttr` (e.g. `[D0, 42, N]`) with static rank;
        dimensions may be integers or parametric. The element type is an M::DType,
        with `invalid` meaning unknown.
    
        Examples:
        ```mlir
        !mo.tensor<[4, 16], f32>           // static shape
        !mo.tensor<[N, N, 6], i32>         // parameterized shape
        !mo.tensor<[1, M, N], i32>         // partially known and parameterized shape
        !mo.tensor<[4, 16], f32, gpu:0>    // optional device
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::MO::TensorType>(), &typeid(M::MO::TensorType));
