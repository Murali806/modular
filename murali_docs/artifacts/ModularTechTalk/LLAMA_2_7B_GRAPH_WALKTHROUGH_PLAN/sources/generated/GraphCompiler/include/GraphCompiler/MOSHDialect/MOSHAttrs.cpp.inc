/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::M::MOSH::ShapeAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::M::MOSH::ShapeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::M::MOSH::ShapeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::M::MOSH::ShapeAttr>([&](auto t) {
      printer << ::M::MOSH::ShapeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace M::MOSH {

namespace detail {

struct ShapeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<TypedAttr>, MOSH::ShapeType>;
  ShapeAttrStorage(::llvm::ArrayRef<TypedAttr> values, MOSH::ShapeType type) : values(std::move(values)), type(std::move(type)) {}

  KeyTy getAsKey() const {
    return KeyTy(values, type);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (::llvm::ArrayRef<TypedAttr>(values) == ::llvm::ArrayRef<TypedAttr>(std::get<0>(tblgenKey))) && (type == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static ShapeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto values = std::move(std::get<0>(tblgenKey));
    auto type = std::move(std::get<1>(tblgenKey));
    values = allocator.copyInto(values);
    return new (allocator.allocate<ShapeAttrStorage>()) ShapeAttrStorage(std::move(values), std::move(type));
  }

  ::llvm::ArrayRef<TypedAttr> values;
  MOSH::ShapeType type;
};

} // namespace detail
ShapeAttr ShapeAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<TypedAttr> values, MOSH::ShapeType type) {
  return Base::get(context, std::move(values), std::move(type));
}

ShapeAttr ShapeAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<TypedAttr> values, MOSH::ShapeType type) {
  return Base::getChecked(emitError, context, std::move(values), std::move(type));
}

ShapeAttr ShapeAttr::get(ArrayRef<TypedAttr> values, MOSH::ShapeType type) {
  return Base::get(type.getContext(), values, type);
}

ShapeAttr ShapeAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ArrayRef<TypedAttr> values, MOSH::ShapeType type) {
  return Base::getChecked(emitError, type.getContext(), values, type);
}

::llvm::LogicalResult ShapeAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<TypedAttr> values, MOSH::ShapeType type) {
  if (::mlir::failed(verify(emitError, values, type)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute ShapeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<TypedAttr>> _result_values;
  ::mlir::FailureOr<MOSH::ShapeType> _result_type;

  if (odsType) {
    if (auto reqType = ::llvm::dyn_cast<MOSH::ShapeType>(odsType)) {
      _result_type = reqType;
    } else {
      odsParser.emitError(odsLoc, "invalid kind of type specified");
      return {};
    }
  } else {
    odsParser.emitError(odsLoc, "expected a trailing type");
    return {};
  }
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};
  // Parse literal ']'
  if (odsParser.parseOptionalRSquare()) {
    {
      auto odsCustomLoc = odsParser.getCurrentLocation();
      (void)odsCustomLoc;
      auto odsCustomResult = parseShapeValue(odsParser,
        ::mlir::detail::unwrapForCustomParse(_result_values),
        *_result_type);
      if (::mlir::failed(odsCustomResult)) return {};
    }
    // Parse literal ']'
    if (odsParser.parseRSquare()) return {};
  } else {
  }
  assert(::mlir::succeeded(_result_type));
  return odsParser.getChecked<ShapeAttr>(odsLoc, odsParser.getContext(),
      ::llvm::ArrayRef<TypedAttr>((_result_values.value_or(::llvm::SmallVector<TypedAttr>()))),
      MOSH::ShapeType((*_result_type)));
}

void ShapeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "[";
  if (!(!(::llvm::ArrayRef<TypedAttr>(getValues()) == ::llvm::ArrayRef<TypedAttr>(::llvm::SmallVector<TypedAttr>())))) {
    odsPrinter << "]";
  } else {
    printShapeValue(odsPrinter,
      getValues(),
      getType());
    odsPrinter << "]";
  }
}

::llvm::ArrayRef<TypedAttr> ShapeAttr::getValues() const {
  return getImpl()->values;
}

MOSH::ShapeType ShapeAttr::getType() const {
  return getImpl()->type;
}


} // namespace M::MOSH
MLIR_DEFINE_EXPLICIT_TYPE_ID(::M::MOSH::ShapeAttr)
namespace M::MOSH {


/// Parse an attribute registered to this dialect.
::mlir::Attribute MOSHDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void MOSHDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace M::MOSH

#endif // GET_ATTRDEF_CLASSES

