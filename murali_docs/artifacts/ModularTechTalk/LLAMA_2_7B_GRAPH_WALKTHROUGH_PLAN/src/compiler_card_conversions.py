"""Source-derived, schematic object/IR construction journeys for explanation cards."""
import html
import json

from compiler_capi_conversion import render_capi_conversion
from compiler_kernel_validation import render_kernel_validation

# Each row is a concrete construction action and the artifact retained afterward.
# The input packet moves through the workbench while new records accumulate.
STORIES = {
'context': ('Python context selection', ['Graph G1', 'calling thread'], 'Active context', [
('Read shared context', '_DEFAULT_MLIR_CONTEXT', 'Shared context is a module-level object.'),
('Activate on this thread', 'current MLIR Context = default', 'The decorator enters a thread-local context scope.'),
('Build IR in that scope', 'G1 types and operations use Context', 'Context supplies ownership for IR construction.'),
('Leave the scope', 'previous context restored', 'The shared default object survives scope exit.')]),
'package': ('Mojo package preparation', ['kernels/*.mojo', 'registered AddOne'], 'Package P1', [
('Resolve source package', 'kernels/__init__.mojo + modules', 'Python resolves the kernel directory.'),
('Run mojo precompile', 'parsed and checked package IR', 'A separate compiler process parses and checks Mojo source.'),
('Serialize package', 'P1: .mojoc bytecode', 'The package retains parametric implementation IR.'),
('Attach package paths', 'G1._kernel_library_paths = [P1]', 'MO call recording can now refer to the registered implementation.')]),
'location': ('Python stack → IR provenance', ['Python call frames', 'capture/profile settings'], 'Location metadata', [
('Choose capture policy', 'unknown or captured source Location', 'Disabled traceback capture uses Location.unknown().'),
('Create loc', 'MLIR Location handle', 'Captured locations describe Python source provenance.'),
('Attach loc to module', 'builtin.module location = loc', 'The module operation retains this metadata.'),
('Attach loc to graph inputs', 'mo.graph + block-argument locations', 'GraphOp uses its location when adding input arguments.')]),
'module-setup': ('Python module choice → MLIR container', ['module=None or existing', 'loc'], 'MLIR module', [
('Select branch', 'existing module or new container', 'The module argument determines reuse versus creation.'),
('Wrap/create builtin.ModuleOp', 'MODULE1: builtin.module', 'Reuse retains the same IR; creation produces an empty body.'),
('Select body block', 'MODULE1.body', 'The body contains top-level operations.'),
('Retain container', 'G1._module → MODULE1', 'The graph will be inserted into this module.')]),
'op-builder': ('Python body.end → native insertion point', ['MODULE1.body', 'end iterator'], 'Builder position', [
('Get block end', 'InsertPoint(block, end)', 'The insertion point follows existing module operations.'),
('Create OpBuilder', 'builder at MODULE1.body.end', 'The native builder retains block and iterator.'),
('Create graph operation', 'mo.graph @addition', 'The graph becomes a module-body operation.'),
('Retain siblings', 'encoder / addition are siblings', 'Insertion does not replace an earlier graph.')]),
'type-conversion': ('Python TensorType → native MO type', ['float32 + [5,10]', 'cpu:0 / gpu:0'], 'MO tensor type', [
('Convert shape', 'ShapeAttr: typed sizes 5 and 10', 'IntegerAttr(si64, size) is wrapped as a KGEN dimension expression.'),
('Retain dtype', 'dtype field = float32', 'The size attributes do not change float32 tensor elements.'),
('Convert device + metadata', 'DeviceRefAttr + DictionaryAttr', 'Add-one has empty layout metadata.'),
('Construct native tensor type', 'T: !mo.tensor<[5,10], f32>', 'T describes an input/result; it contains no tensor numbers.')]),
'native-builder': ('Python GraphOp arguments → MO graph shell', ['inputTypes=[T]', 'resultTypes=[]'], 'Graph G1 body', [
('Record signature', 'signature: (T) → ()', 'The initial output type list is empty.'),
('Create body region', 'G1.region[0]', 'The builder creates the region.'),
('Create ONE entry block', 'G1.region[0].block[0]', 'All inputs belong to the same block.'),
('Append typed argument', '%x : T in entry block', 'Each input type adds an argument, not another block.')]),
'symbolic-params': ('Python shape names → KGEN declarations', ['shape=[batch,10]', 'static case=[5,10]'], 'Input parameters', [
('Discover SymbolicDim names', 'names = [batch] or []', 'Static sizes do not become parameter names.'),
('Deduplicate names', '_params = {batch: None} or {}', 'Repeated batch references share one declaration.'),
('Assign parameter type', 'ParamDeclAttr(batch, si64)', 'The size parameter is a signed 64-bit value.'),
('Attach array to GraphOp', 'input_parameters = [batch] or []', 'Add-one attaches an empty declaration array.')]),
'retain-body': ('Native operation → Python retained handles', ['op._CAPIPtr', 'existing graph region'], 'G1 references', [
('Bridge existing operation', 'G1._mlir_op → mo.graph', 'This creates a wrapper, not a graph clone.'),
('Select region and block', 'regions[0].blocks[0]', 'The native builder already created this block.'),
('Retain current target', '_current_block → entry block', 'Later operations use the current insertion target.'),
('Retain original graph body', '_graph_body → same entry block', 'Graph.inputs reads this block’s arguments.')]),
'device-info': ('Driver descriptions → MLIR module attributes', ['CPU() metadata', 'optional Accelerator()'], 'Module target map', [
('Check existing mapping', 'reuse or create mapping', 'An existing module mapping causes an early return.'),
('Read device descriptions', 'label / api / arch / model', 'Missing architecture or model queries use unknown.'),
('Build DeviceInfoAttr entries', 'cpu:0 + default accelerator', 'This helper adds at most one default accelerator.'),
('Attach dictionary', 'module.mo.device_info_mapping', 'No kernel is compiled or executed here.')]),
 'tracking': ('Python bookkeeping → symbolic registries', ['G1', 'no weights yet'], 'Graph state', [
('Initialize weights', '_weights = {}', 'This holds graph declarations, not runtime tensor storage.'),
('Initialize chain flag', '_has_chain_input = False', 'Incoming ordering has not been detected yet.'),
('Create lazy map', 'device_chains = {}', 'The map retains G1 for later insertion.'),
('Register device on lookup', 'cpu:0 → CH0; gpu:0 → seed', 'This last view illustrates lazy lookup after CPU seeding.')]),
'ready-chain': ('Python ChainCreate call → MO dependency seed', ['inputs=[]', 'result=ChainType()'], 'Chain references', [
('Choose graph insertion block', 'OpBuilder at current block end', 'Generated operations use the graph insertion target.'),
('Insert chain operation', 'CH0 = mo.chain.create([])', 'Empty inputs mean no producer dependencies.'),
('Wrap result', '_always_ready_chain → CH0', 'The graph retains this unadvanced seed.'),
('Seed host ordering map', 'device_chains[cpu:0] → CH0', 'Future ordered effects can advance the map separately.')]),
 'trailing-chain': ('Input type check → adopted ordering value', ['last body argument', '%x or %incoming'], 'Host ordering', [
('Read final argument', '%x:T or %incoming:chain', 'The constructor only inspects the existing last input.'),
('Test native type', '_is_chain_value = False / True', 'A tensor input is not an ordering token.'),
('Adopt only if chain', '_has_chain_input = True if chain', 'Add-one takes the false branch.'),
('Select host chain', 'CH0 for add-one / incoming for child', 'The always-ready seed remains retained in either case.')]),
'forward-context': ('Python with G1 → active IR recording scope', ['G1 Graph object', 'previous graph context'], 'Active scope', [
('Enter context manager', 'Graph.__enter__ → _enter', 'Python invokes the with protocol.'),
('Ensure MLIR context', 'IR construction context active', 'This is needed on the calling thread.'),
('Set graph ContextVar', 'CURRENT_GRAPH = G1', 'ops.custom can resolve its destination graph.'),
('Exit after recording', 'previous CURRENT_GRAPH restored', 'The final view represents leaving the with scope.')]),
'forward-inputs': ('Native input argument → Python symbolic x', ['%arg0:T in G1', 'optional chain arguments'], 'Forward inputs', [
('Read body arguments', 'arguments = [%arg0]', 'These are IR values, not input buffers.'),
('Omit trailing internal chains', 'user arguments = [%arg0]', 'The forward callback receives only user values.'),
('Create Python wrappers', 'TensorValue → %arg0', 'The native input stays in G1.'),
('Expand callback arguments', 'forward(x)', 'x refers to the same symbolic input.')]),
'forward-callback': ('Python lambda → declared custom result type', ['symbolic x:T', 'selected device'], 'Result type', [
('Run graph-building lambda', 'forward(x)', 'Python executes now; Mojo arithmetic does not.'),
('Read x metadata', 'x.dtype=float32; shape=[5,10]', 'Only symbolic type properties are read.'),
('Declare Python output type', 'out_types=[TensorType(...)]', 'No output data buffer is allocated.'),
('Convert type for custom call', 'result type T in MO call', 'The call produces a future symbolic value of T.')]),
'custom-attributes': ('Python call arguments → MO call attributes', ['name=add_one', 'device + out_types'], 'Call fields', [
('Resolve active G1', 'graph = Graph.current', 'The context supplies the insertion destination.'),
('Convert kernel name', 'StringAttr.get("add_one", context)', 'The name refers to the registration in P1.'),
('Convert declared output type', 'result_types = [float32[5,10] cpu:0]', 'This tells the call its future result type.'),
('Attach device after insertion', 'device = m.DeviceRefAttr("cpu", 0)', 'Kernel verification follows device attachment.')]),
'custom-insert': ('Python ops.custom → recorded MO call', ['x → %arg0', 'P1 registration add_one'], 'G1 operation body', [
('Unwrap Python handles', '%arg0 : float32[5,10] on cpu:0', 'The existing symbolic input is referenced.'),
('Select insertion point', 'G1.body insertion point', 'The graph determines where the call is placed.'),
('Create new MO operation', 'CUST1: mo.custom(add_one, %arg0)', 'The Mojo body remains in P1.'),
('Retain symbolic result', '%r : float32[5,10] on cpu:0', 'No matrix values have been calculated.')]),
'custom-verify': ('MO call contract ↔ Mojo definitions', ['CUST1 input/result types', 'P1 AddOne declaration'], 'Validated call', [
('Read package paths', 'Analysis key = package paths', 'Analysis may be reused for the same packages.'),
('Resolve registered declaration', 'add_one → AddOne.execute', 'The registered Mojo definition supplies the contract.'),
('Verify symbolic call', 'device / arguments / results checked', 'Invalid calls raise; this illustrates the valid route.'),
('Return validated result', 'CUST1 retained in G1', 'Validation does not execute or inline Mojo arithmetic.')]),
'custom-result': ('MO result → Python list → callback return', ['%r:T', 'out_types has one type'], 'Result wrappers', [
('Read native result', '%r owned by CUST1', 'The result is symbolic.'),
('Wrap as Python value', 'TensorValue → %r', 'The wrapper refers to the same native result.'),
('Return results list', '[Y1: TensorValue referring to %r]', 'One wrapper per declared output type.'),
('Select first tensor result', 'results[0].tensor → constructor', 'Selection does not copy or execute a tensor.')]),
'forward-output': ('Python result → completed MO graph', ['symbolic %r:T', 'input %x:T'], 'Completed G1', [
('Collect output value and type', 'outputs=[%r]; types=[T]', 'The output contract is inferred from symbolic results.'),
('Update graph signature', '(T) → (T)', 'The shell’s empty result list becomes one output.'),
('Insert output and verify', 'mo.output(%r); G1 verified', 'The output operation completes the graph body.'),
('Leave recording scope', 'G1 retained; context restored', 'Compile/init/execute follow later.')]),
'objects': ('Python Graph → nested MO graph definition', ['name=addition', 'input type T'], 'Module structure', [
('Create/reuse module', 'builtin.module', 'This is the enclosing IR container.'),
('Insert graph definition', 'mo.graph @addition', 'The graph owns its body region.'),
('Record custom operation', '%r = custom call using %x', 'Python builds symbolic MO operations.'),
('Record output', 'mo.output(%r); (T) → (T)', 'Mojo implementation remains a separate package.')]),
'validation': ('Package P1 + Python call → verified G1', ['P1 + symbolic %x', 'declared result T'], 'Verified MO graph', [
('Resolve custom registration', 'add_one → P1 declaration', 'The implementation package has already been prepared.'),
('Record mo.custom', 'CUST1 operand=%x, result=%r', 'A call appears in the graph body.'),
('Verify kernel contract', 'CUST1 contract checked', 'This example supplies output types explicitly.'),
('Finish Graph.output', 'G1: verified MO graph', 'G1 is ready to be supplied to compile.')]),
}

# Session/compile/init cards explain later phases, rather than claiming Python→MO there.
LATER = {
'session-ownership': ('Device handles → session S1', ['Device handles','constructor options'], 'Session ownership', ['N1.devices=[CPU0]; CFG1 paths=[]', 'runtime: threads=0; API=python', 'N1 owns E1; E1 → shared CX1', 'CX1 retains HRT1 + typed services']),
'compile-passes': ('G1 clone → lowered graph representations', ['verified G1','P1'], 'Compiler graph C1', ['clone retained separately from G1', 'MO → MAP', 'MAP → MOGG', 'MOGG → MGP execution plan']),
'compile-kernels': ('Mojo specialization → target kernel code', ['P1 implementations','target metadata'], 'Compiled kernels', ['specialized kernel IR', 'MLIR LLVM dialect', 'translated LLVM IR module', 'host/device target artifacts']),
'compile-mef': ('Compiled graph and code → CM1 bytes', ['MGP plan','compiled target code'], 'Prepared artifact', ['compiled routines retained', 'model descriptors prepared', 'MEF archive assembled', 'CM1: uninitialized compiled bytes']),
'init-resources': ('CM1 → initialized model M1', ['CM1 descriptors','S1 device context'], 'Runtime resources', ['map device references', 'load compiled routines', 'invoke initialization callable', 'M1 executable handles']),
'init-gates': ('Initialization prerequisites → ready M1', ['compiled CM1','weight registry'], 'Readiness state', ['resolve compiled artifact', 'validate required bindings', 'await setup completion', 'M1 ready; retained ownership']),
'example': ('Actual tensor data → execute → result', ['B1: real input buffer','initialized M1'], 'Execution buffers', ['submit input B1 to M1', 'invoke selected target code', 'complete computation', 'O1: input values plus one']),
}


def render_card_conversion(key, source_key, refs):
    if key == "custom-verify":
        return render_kernel_validation(refs)
    if key == "retain-body":
        return render_capi_conversion(refs)
    story = STORIES.get(key)
    if story is None:
        title, inputs, destination, records = LATER[key]
        story = (title, inputs, destination, [(f'Step {i+1}', record, 'Source-derived phase illustration; no measured execution.') for i, record in enumerate(records)])
    title, inputs, destination, steps = story
    source_key = {
        "context": "ct_graph_context", "package": "ct_precompile",
        "location": "ct_graph_location", "type-conversion": "ct_tensor_to_mlir",
        "native-builder": "ct_graph_op_build", "device-info": "ct_graph_device",
        "tracking": "ct_device_chain_map", "ready-chain": "ct_generated_op",
        "trailing-chain": "ct_is_chain", "forward-context": "ct_graph_enter",
        "forward-inputs": "ct_graph_inputs", "forward-callback": "entry",
        "custom-attributes": "custom_api", "custom-insert": "ct_insert_result",
        "custom-verify": "ct_graph_verify", "custom-result": "ct_insert_result",
        "forward-output": "graph_output",
    }.get(key, source_key)
    r = refs[source_key]
    frames = [dict(title='Inputs ready', record='', detail='Step through the construction. Each record remains visible as the next is added.', packet=' + '.join(inputs), index=0)]
    for i, (action, record, detail) in enumerate(steps, 1):
        frames.append(dict(title=action, record=record, detail=detail, packet=record, index=i))
    payload = html.escape(json.dumps(frames, ensure_ascii=False), quote=True)
    records = ''.join(f'<g data-conversion-record="{i}" opacity="0"><rect x="334" y="{70+i*33}" width="350" height="28" rx="4"/><text x="342" y="{89+i*33}">{html.escape(record)}</text></g>' for i, (_, record, _) in enumerate(steps))
    transcript = ''.join(f'<li><b>{html.escape(action)}</b> → <code>{html.escape(record)}</code>. {html.escape(detail)}</li>' for action, record, detail in steps)
    return (f'<section class="card-conversion" data-card-conversion="{key}" data-conversion-frames="{payload}" data-conversion-index="0" aria-label="{html.escape(title)}">'
        f'<h3>{html.escape(title)}</h3><p class="conversion-evidence"><b>SOURCE</b> · Schematic objects and IR; no compiler dump or measured timing.</p>'
        '<div class="conversion-controls"><button type="button" data-conversion-action="back">Back</button><button type="button" data-conversion-action="step">Step</button><button type="button" data-conversion-action="play">Play</button><button type="button" data-conversion-action="reset">Reset</button><output data-conversion-count>0 / 4</output></div>'
        f'<div class="conversion-canvas"><svg viewBox="0 0 720 290" role="img" aria-label="{html.escape(title)}; accumulating construction records">'
        '<rect class="conversion-source" x="14" y="20" width="275" height="65" rx="8"/>'
        '<text x="25" y="43">SOURCE INPUTS · retained</text>'
        f'<text x="25" y="61">{html.escape(inputs[0])}</text>'
        f'<text x="25" y="78">{html.escape(inputs[1])}</text>'
        '<rect class="conversion-container" x="322" y="20" width="375" height="240" rx="8"/>'
        f'<text x="336" y="43">{html.escape(destination)} · grows step by step</text>'
        '<path class="conversion-route" d="M150 90 V240 H306 V70 H330"/>'
        + records + '<g data-conversion-packet transform="translate(22,110)"><path d="M0 0 H244 L258 14 V60 H0 Z"/><text x="10" y="22">INPUT / NEW RECORD</text><foreignObject x="9" y="28" width="240" height="30"><div xmlns="http://www.w3.org/1999/xhtml" data-conversion-packet-label>Inputs ready</div></foreignObject></g>'
        '</svg></div><p class="conversion-current" aria-live="polite"><b data-conversion-title>Inputs ready</b><span data-conversion-detail>Use Step or Play.</span></p>'
        f'<a data-ref="{source_key}" data-line="{r["line"]}" href="../sources/{html.escape(r["file"])}#L{r["line"]}">Open source · {html.escape(r["label"])}</a>'
        f'<details class="conversion-transcript"><summary>Portable conversion steps</summary><ol>{transcript}</ol></details></section>')
