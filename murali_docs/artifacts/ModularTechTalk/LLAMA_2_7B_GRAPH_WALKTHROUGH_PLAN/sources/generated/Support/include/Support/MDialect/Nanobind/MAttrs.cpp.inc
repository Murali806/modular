nb::class_<M::ArrayElementsAttr, ::mlir::Attribute>(m, "ArrayElementsAttr")
    .def(nb::new_(&get_checked<M::ArrayElementsAttr, ArrayRef<uint8_t>, ShapedType>), nb::arg("data"), nb::arg("type"))
    .def(nb::new_(infer_context(&get_checked<M::ArrayElementsAttr, mlir::MLIRContext*, PrimitiveArrayAttr, ShapedType>)), nb::arg("data"), nb::arg("type"))
    .def_prop_ro("data", optional_pimpl(&M::ArrayElementsAttr::getData))
    .def_prop_ro("type", optional_pimpl(&M::ArrayElementsAttr::getType))
    .doc() = R"nbdoc(
        The `#M.dense_array` attribute is an elements attribute backed by a
        primitive array. The attribute supports the full API expected by
        `ElementsAttr` and users thereof, including arbitrary shaped types and
        failable value iteration.
    
        Importantly, this attribute can only contain scalar integer and floating
        point types, does not bit-pack elements, and does not have special handling
        for splat elements.
    
        This attribute implements the HasAlignedBytesInterface by returning a
        !M.aligned_bytes type capturing the byte size of the underlying element data
        array and an alignment of the next power-of-two of the byte size of the
        element type.
    
        Example:
    
        ```mlir
        // An array of integers.
        #M.dense_array<-3, 0, 1, 42> : !M.array<4xi32>
    
        // A float tensor.
        #M.dense_array<3.4, 2.3, 5.2, 1.9> : tensor<2x2xf32>
    
        // A 0D vector.
        #M.dense_array<1> : vector<ui64>
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::ArrayElementsAttr>(), &typeid(M::ArrayElementsAttr));
nb::class_<M::DeviceInfoAttr, ::mlir::Attribute>(m, "DeviceInfoAttr")
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, llvm::StringRef, llvm::StringRef, llvm::StringRef, llvm::StringRef, bool>(&M::DeviceInfoAttr::get))), nb::arg("label"), nb::arg("api"), nb::arg("arch"), nb::arg("model"), nb::arg("tile_based_fusion"))
    .def_prop_ro("label", optional_pimpl(&M::DeviceInfoAttr::getLabel))
    .def_prop_ro("api", optional_pimpl(&M::DeviceInfoAttr::getApi))
    .def_prop_ro("arch", optional_pimpl(&M::DeviceInfoAttr::getArch))
    .def_prop_ro("model", optional_pimpl(&M::DeviceInfoAttr::getModel))
    .def_prop_ro("tile_based_fusion", optional_pimpl(&M::DeviceInfoAttr::getTileBasedFusion))
    .doc() = R"nbdoc(
        The `#M.device_info` attribute captures runtime-probed properties of a
        physical device: its label (e.g. `"gpu"`), the underlying API
        (e.g. `"cuda"`, `"hip"`), the architecture name
        (e.g. `"sm_100a"`, `"gfx942"`), and the model name
        (e.g. `"NVIDIA B200"`). This is used to drive per-kernel device
        registration and specialization.
    
        Example:
        ```mlir
          #M.device_info<"gpu", "cuda", "sm_100a", "NVIDIA B200">
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::DeviceInfoAttr>(), &typeid(M::DeviceInfoAttr));
nb::class_<M::DeviceRefAttr, ::mlir::Attribute>(m, "DeviceRefAttr")
    .def(nb::new_(infer_context(nb::overload_cast<mlir::MLIRContext*, llvm::StringRef, DeviceId>(&M::DeviceRefAttr::get))), nb::arg("label"), nb::arg("id"))
    .def_prop_ro("label", optional_pimpl(&M::DeviceRefAttr::getLabel))
    .def_prop_ro("id", optional_pimpl(&M::DeviceRefAttr::getId))
    .doc() = R"nbdoc(
        The `#M.device_ref` attribute refers to a unique `#M.device_spec` within the
        overall model `#M.device_spec_collection`. It contain a label and id.
    
        Example:
        ```mlir
          #M.device_ref<"gpu", 0>
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::DeviceRefAttr>(), &typeid(M::DeviceRefAttr));
nb::class_<M::PrimitiveArrayAttr, ::mlir::Attribute>(m, "PrimitiveArrayAttr")
    .def(nb::new_(&get_checked<M::PrimitiveArrayAttr, ArrayRef<uint8_t>, Type>), nb::arg("data"), nb::arg("element_type"))
    .def(nb::new_(infer_context(&get_checked<M::PrimitiveArrayAttr, mlir::MLIRContext*, llvm::ArrayRef<uint8_t>, Type>)), nb::arg("data"), nb::arg("element_type"))
    .def_prop_ro("data", optional_pimpl(&M::PrimitiveArrayAttr::getData))
    .def_prop_ro("element_type", optional_pimpl(&M::PrimitiveArrayAttr::getElementType))
    .doc() = R"nbdoc(
        The `#M.primitives_array` attribute represents an array of primitive
        (boolean, integer, index, or floating point) data of equal size. The data is
        stored as a byte array with element types whose sizes are not multiples of
        bytes padded to the nearest byte. The underlying array is aligned to that
        byte size.
    
        Example:
    
        ```mlir
        // An array of integers.
        #M.primitives_array<si24: -2, 0, 2>
    
        // An array of floats.
        #M.primitives_array<bf16: 0.2, 1.2, 3.>
    
        // Boolean arrays use `i1` elements.
        #M.primitives_array<i1: true, false, true>
        ```
      )nbdoc";
    ::M::Graph::Python::registerTypeID(::mlir::TypeID::get<M::PrimitiveArrayAttr>(), &typeid(M::PrimitiveArrayAttr));
