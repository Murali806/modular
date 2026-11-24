/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::M::MO::TensorType,
::M::MO::ListType,
::M::MO::BundleType,
::M::MO::ScalarType,
::M::MO::BufferType,
::M::MO::ChainType,
::M::MO::OpaqueType

#endif // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::M::MO::TensorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::M::MO::TensorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::M::MO::ListType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::M::MO::ListType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::M::MO::BundleType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::M::MO::BundleType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::M::MO::ScalarType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::M::MO::ScalarType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::M::MO::BufferType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::M::MO::BufferType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::M::MO::ChainType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::M::MO::ChainType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::M::MO::OpaqueType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::M::MO::OpaqueType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::M::MO::TensorType>([&](auto t) {
      printer << ::M::MO::TensorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::M::MO::ListType>([&](auto t) {
      printer << ::M::MO::ListType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::M::MO::BundleType>([&](auto t) {
      printer << ::M::MO::BundleType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::M::MO::ScalarType>([&](auto t) {
      printer << ::M::MO::ScalarType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::M::MO::BufferType>([&](auto t) {
      printer << ::M::MO::BufferType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::M::MO::ChainType>([&](auto t) {
      printer << ::M::MO::ChainType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::M::MO::OpaqueType>([&](auto t) {
      printer << ::M::MO::OpaqueType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace M::MO {

namespace detail {

struct TensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::M::MOSH::ShapeAttr, DType, ::M::DeviceRefAttr, DictionaryAttr>;
  TensorTypeStorage(::M::MOSH::ShapeAttr shape, DType dtype, ::M::DeviceRefAttr deviceRef, DictionaryAttr metadata) : shape(std::move(shape)), dtype(std::move(dtype)), deviceRef(std::move(deviceRef)), metadata(std::move(metadata)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, dtype, deviceRef, metadata);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (dtype == std::get<1>(tblgenKey)) && (deviceRef == std::get<2>(tblgenKey)) && (metadata == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static TensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto dtype = std::move(std::get<1>(tblgenKey));
    auto deviceRef = std::move(std::get<2>(tblgenKey));
    auto metadata = std::move(std::get<3>(tblgenKey));
    return new (allocator.allocate<TensorTypeStorage>()) TensorTypeStorage(std::move(shape), std::move(dtype), std::move(deviceRef), std::move(metadata));
  }

  ::M::MOSH::ShapeAttr shape;
  DType dtype;
  ::M::DeviceRefAttr deviceRef;
  DictionaryAttr metadata;
};

} // namespace detail
::llvm::LogicalResult TensorType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::M::MOSH::ShapeAttr shape, DType dtype, ::M::DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  if (::mlir::failed(verify(emitError, shape, dtype, deviceRef, metadata)))
    return ::mlir::failure();
  return ::mlir::success();
}

::M::MOSH::ShapeAttr TensorType::getShape() const {
  return getImpl()->shape;
}

DType TensorType::getDtype() const {
  return getImpl()->dtype;
}

::M::DeviceRefAttr TensorType::getDeviceRef() const {
  return getImpl()->deviceRef;
}

DictionaryAttr TensorType::getMetadata() const {
  return getImpl()->metadata;
}


} // namespace M::MO
MLIR_DEFINE_EXPLICIT_TYPE_ID(::M::MO::TensorType)
namespace M::MO {

namespace detail {

struct ListTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  ListTypeStorage(Type elementType) : elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ListTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ListTypeStorage>()) ListTypeStorage(std::move(elementType));
  }

  Type elementType;
};

} // namespace detail
ListType ListType::get(::mlir::MLIRContext *context, Type elementType) {
  return Base::get(context, std::move(elementType));
}

ListType ListType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType) {
  return Base::getChecked(emitError, context, std::move(elementType));
}

::llvm::LogicalResult ListType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType) {
  if (::mlir::failed(verify(emitError, elementType)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type ListType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'elementType'
  _result_elementType = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_elementType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MO_List parameter 'elementType' which is to be a `Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  return odsParser.getChecked<ListType>(odsLoc, odsParser.getContext(),
      Type((*_result_elementType)));
}

void ListType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getElementType());
  odsPrinter << ">";
}

Type ListType::getElementType() const {
  return getImpl()->elementType;
}


} // namespace M::MO
MLIR_DEFINE_EXPLICIT_TYPE_ID(::M::MO::ListType)
namespace M::MO {

namespace detail {

struct BundleTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<Type>>;
  BundleTypeStorage(::llvm::ArrayRef<Type> elementTypes) : elementTypes(std::move(elementTypes)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementTypes);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementTypes == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static BundleTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementTypes = std::move(std::get<0>(tblgenKey));
    elementTypes = allocator.copyInto(elementTypes);
    return new (allocator.allocate<BundleTypeStorage>()) BundleTypeStorage(std::move(elementTypes));
  }

  ::llvm::ArrayRef<Type> elementTypes;
};

} // namespace detail
BundleType BundleType::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Type> elementTypes) {
  return Base::get(context, std::move(elementTypes));
}

BundleType BundleType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<Type> elementTypes) {
  return Base::getChecked(emitError, context, std::move(elementTypes));
}

::llvm::LogicalResult BundleType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<Type> elementTypes) {
  if (::mlir::failed(verify(emitError, elementTypes)))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::ArrayRef<Type> BundleType::getElementTypes() const {
  return getImpl()->elementTypes;
}


} // namespace M::MO
MLIR_DEFINE_EXPLICIT_TYPE_ID(::M::MO::BundleType)
namespace M::MO {

namespace detail {

struct ScalarTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<DType>;
  ScalarTypeStorage(DType dtype) : dtype(std::move(dtype)) {}

  KeyTy getAsKey() const {
    return KeyTy(dtype);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (dtype == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ScalarTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto dtype = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ScalarTypeStorage>()) ScalarTypeStorage(std::move(dtype));
  }

  DType dtype;
};

} // namespace detail
ScalarType ScalarType::get(::mlir::MLIRContext *context, DType dtype) {
  return Base::get(context, std::move(dtype));
}

::mlir::Type ScalarType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<DType> _result_dtype;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'dtype'
  _result_dtype = ::mlir::FieldParser<DType>::parse(odsParser);
  if (::mlir::failed(_result_dtype)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MO_Scalar parameter 'dtype' which is to be a `DType`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_dtype));
  return ScalarType::get(odsParser.getContext(),
      DType((*_result_dtype)));
}

void ScalarType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getDtype());
  odsPrinter << ">";
}

DType ScalarType::getDtype() const {
  return getImpl()->dtype;
}


} // namespace M::MO
MLIR_DEFINE_EXPLICIT_TYPE_ID(::M::MO::ScalarType)
namespace M::MO {

namespace detail {

struct BufferTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::M::MOSH::ShapeAttr, DType, ::M::DeviceRefAttr, DictionaryAttr>;
  BufferTypeStorage(::M::MOSH::ShapeAttr shape, DType dtype, ::M::DeviceRefAttr deviceRef, DictionaryAttr metadata) : shape(std::move(shape)), dtype(std::move(dtype)), deviceRef(std::move(deviceRef)), metadata(std::move(metadata)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, dtype, deviceRef, metadata);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (dtype == std::get<1>(tblgenKey)) && (deviceRef == std::get<2>(tblgenKey)) && (metadata == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static BufferTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto dtype = std::move(std::get<1>(tblgenKey));
    auto deviceRef = std::move(std::get<2>(tblgenKey));
    auto metadata = std::move(std::get<3>(tblgenKey));
    return new (allocator.allocate<BufferTypeStorage>()) BufferTypeStorage(std::move(shape), std::move(dtype), std::move(deviceRef), std::move(metadata));
  }

  ::M::MOSH::ShapeAttr shape;
  DType dtype;
  ::M::DeviceRefAttr deviceRef;
  DictionaryAttr metadata;
};

} // namespace detail
::llvm::LogicalResult BufferType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::M::MOSH::ShapeAttr shape, DType dtype, ::M::DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  if (::mlir::failed(verify(emitError, shape, dtype, deviceRef, metadata)))
    return ::mlir::failure();
  return ::mlir::success();
}

::M::MOSH::ShapeAttr BufferType::getShape() const {
  return getImpl()->shape;
}

DType BufferType::getDtype() const {
  return getImpl()->dtype;
}

::M::DeviceRefAttr BufferType::getDeviceRef() const {
  return getImpl()->deviceRef;
}

DictionaryAttr BufferType::getMetadata() const {
  return getImpl()->metadata;
}


} // namespace M::MO
MLIR_DEFINE_EXPLICIT_TYPE_ID(::M::MO::BufferType)
namespace M::MO {


} // namespace M::MO
MLIR_DEFINE_EXPLICIT_TYPE_ID(::M::MO::ChainType)
namespace M::MO {

namespace detail {

struct OpaqueTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<StringAttr, ::mlir::DictionaryAttr>;
  OpaqueTypeStorage(StringAttr symbol, ::mlir::DictionaryAttr parameters) : symbol(std::move(symbol)), parameters(std::move(parameters)) {}

  KeyTy getAsKey() const {
    return KeyTy(symbol, parameters);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (symbol == std::get<0>(tblgenKey)) && (parameters == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static OpaqueTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto symbol = std::move(std::get<0>(tblgenKey));
    auto parameters = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<OpaqueTypeStorage>()) OpaqueTypeStorage(std::move(symbol), std::move(parameters));
  }

  StringAttr symbol;
  ::mlir::DictionaryAttr parameters;
};

} // namespace detail
::mlir::Type OpaqueType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<StringAttr> _result_symbol;
  ::mlir::FailureOr<::mlir::DictionaryAttr> _result_parameters;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'symbol'
  _result_symbol = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_symbol)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MO_Opaque parameter 'symbol' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseOptionalComma()) {
  } else {

    // Parse variable 'parameters'
    _result_parameters = ::mlir::FieldParser<::mlir::DictionaryAttr>::parse(odsParser);
    if (::mlir::failed(_result_parameters)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MO_Opaque parameter 'parameters' which is to be a `::mlir::DictionaryAttr`");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_symbol));
  return OpaqueType::get(odsParser.getContext(),
      StringAttr((*_result_symbol)),
      ::mlir::DictionaryAttr((_result_parameters.value_or(::mlir::DictionaryAttr::get(odsParser.getContext())))));
}

void OpaqueType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getSymbol());
  if (!(getParameters() == ::mlir::DictionaryAttr::get(getContext()))) {
    odsPrinter << ",";
    if (!(getParameters() == ::mlir::DictionaryAttr::get(getContext()))) {
      odsPrinter << ' ';
      odsPrinter.printStrippedAttrOrType(getParameters());
    }
  } else {
  }
  odsPrinter << ">";
}

StringAttr OpaqueType::getSymbol() const {
  return getImpl()->symbol;
}

::mlir::DictionaryAttr OpaqueType::getParameters() const {
  return getImpl()->parameters;
}


} // namespace M::MO
MLIR_DEFINE_EXPLICIT_TYPE_ID(::M::MO::OpaqueType)
namespace M::MO {


/// Parse a type registered to this dialect.
::mlir::Type MODialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void MODialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}

} // namespace M::MO

#endif // GET_TYPEDEF_CLASSES

