"""Concrete symbolic values for the existing forward/custom-call explanation cards."""
FORWARD = '''

**Concrete CPU example · values available while Python records G1**

| Expression | Concrete description | Comes from |
| --- | --- | --- |
| `device` | `CPU()`; symbolic reference `cpu:0` | Example device selection |
| `x` | Python TensorValue X1 referring to native block argument `%arg0` | `self.inputs[0]` |
| `x.dtype` | `DType.float32` | Input type supplied to Graph |
| `x.tensor.shape` | Shape containing static dimensions 5 and 10 | Input shape `[rows, columns]` |
| `DeviceRef.from_device(device)` | CPU reference, id 0 | Same selected device |
| `T_out` | `TensorType(DType.float32, [5,10], device=cpu_ref)` | Output declaration in the lambda |
| `out_types` | `[T_out]` | One expected output |

```python
# Expanded form of the original lambda, for explanation.
# x is a symbolic TensorValue supplied by Graph.__init__.
cpu_ref = DeviceRef.from_device(device)
T_out = TensorType(dtype=x.dtype, shape=x.tensor.shape, device=cpu_ref)
outputs = ops.custom(name="add_one", device=cpu_ref,
                     values=[x], out_types=[T_out])
y = outputs[0].tensor
```

```text
forward(X1) [LINK:graph_init:922]
├─ x.dtype → float32; x.tensor.shape → static [5,10] [LINK:entry:44] [CALLER:entry|43]
├─ DeviceRef.from_device(CPU()) → cpu reference 0 [LINK:entry:46] [CALLER:entry|46]
├─ TensorType(...) → T_out; declares type, allocates no output buffer [LINK:entry:43] [CALLER:entry|43]
├─ ops.custom(..., out_types=[T_out]) → [Y1 referring to %r] [LINK:entry:38] [CALLER:entry|38]
└─ outputs[0].tensor → Y1 → constructor's result [LINK:entry:49] [CALLER:graph_init|922]
```

For a selected accelerator, the same shape/dtype is declared with a GPU device reference (for example `gpu:0`). Neither case puts actual matrix elements in X1 or Y1. Later `model.execute` can receive, for example, a matrix whose first row starts `[0.0, 1.0, 2.0, ...]`; its output then starts `[1.0, 2.0, 3.0, ...]`. Those numbers belong to execution, not this forward callback.
'''
ATTRIBUTES = '''

**Concrete attributes · CPU and GPU examples**

```text
name = "add_one" [LINK:entry:39]
└─ StringAttr.get("add_one", context) [LINK:custom_api:96]
   └─ MLIR string attribute whose value is add_one → symbol field of CUST1 [LINK:custom_api:110] [CALLER:custom_api|96]

device = DeviceRef.from_device(CPU()) [LINK:entry:40]
└─ DeviceRef.from_device(...) → DeviceRef(kind=cpu, id=0) [LINK:custom_api:97]
   └─ DeviceRef.to_mlir() → m.DeviceRefAttr("cpu", 0) [LINK:ct_device_ref_to_mlir:183]
      └─ attribute C API bridge → custom_op.device [LINK:custom_api:122] [CALLER:custom_api|122]

GPU illustration: DeviceRef(kind=gpu, id=0) [LINK:ct_device_ref_to_mlir:183]
└─ m.DeviceRefAttr("gpu", 0) → GPU-labelled call [LINK:ct_device_ref_to_mlir:183] [CALLER:custom_api|122]

out_types = [T_out: float32[5,10] on cpu:0] [LINK:entry:42]
└─ T_out.to_mlir() → native MO tensor type with shape/dtype/device [LINK:ct_tensor_to_mlir:430]
   └─ result_types=[that native type] → one native SSA result %r [LINK:custom_api:110] [CALLER:custom_api|109]
```

The device examples above show constructor arguments and attribute contents, not an asserted textual MLIR printer syntax. A DeviceRefAttr identifies a device kind and index; it does not contain the driver, stream, GPU memory, or compiled kernel. The registration string selects P1's kernel declaration; device metadata guides subsequent compatibility checks.
'''
INSERT = '''

**Complete add-one call · source order and concrete arguments**

```text
ops.custom(name="add_one", device=cpu_ref, values=[X1], out_types=[T_out]) [LINK:entry:38]
├─ Graph.current → active G1; default_mlir_context() → IR context [LINK:custom_api:94] [CALLER:custom_api|94]
├─ symbol_attr = StringAttr.get("add_one", context) [LINK:custom_api:96] [CALLER:custom_api|96]
├─ normalize selected device → CPU DeviceRef, id=0 [LINK:custom_api:97] [CALLER:custom_api|97]
├─ reject Buffer/opaque arguments; normalize tensor-like inputs [LINK:custom_api:99] [CALLER:custom_api|99]
├─ convert T_out → native tensor type float32[5,10] on cpu:0 [LINK:custom_api:110] [CALLER:custom_api|109]
├─ G1._add_op_get_op_with_results(mo.custom, [T_native], [X1], symbol=symbol_attr) [LINK:custom_api:109]
│  ├─ recursively unwrap [X1] → [mlir.Value referring to %arg0] [LINK:ct_insert_result:1299] [CALLER:ct_insert_result|1309]
│  ├─ bridge native type/attribute handles; retain symbol="add_one" [LINK:ct_insert_result:1303] [CALLER:ct_insert_result|1310]
│  ├─ ip = mlir.InsertionPoint(G1._body) [LINK:ct_insert_result:1312] [CALLER:ct_insert_result|1312]
│  ├─ with ip, _location(...): call mo.custom([T_native], [%arg0], symbol=...) [LINK:ct_insert_result:1317] [CALLER:ct_insert_result|1320]
│  ├─ find inserted CUST1; staged_op = staged_op.opview [LINK:ct_insert_result:1323] [CALLER:ct_insert_result|1328]
│  ├─ G1._verify_op(staged_op) → structural IR verification [LINK:ct_insert_result:1329] [CALLER:ct_insert_result|1329]
│  ├─ attach output parameter declarations; wrap native %r as Y1 [LINK:ct_insert_result:1351] [CALLER:ct_insert_result|1360]
│  └─ return ([Y1], CUST1) to ops.custom [LINK:ct_insert_result:1364] [CALLER:custom_api|109]
├─ parameters=None → skip optional parameter dictionary assignment [LINK:custom_api:113] [CALLER:custom_api|113]
├─ attach device attribute m.DeviceRefAttr("cpu",0) to CUST1 [LINK:custom_api:122] [CALLER:custom_api|122]
├─ verify CUST1 against the loaded P1 kernel library [LINK:custom_api:127] [CARD:toolchain-call-graph-custom-verify|Kernel validation · P1 and MO] [CALLER:custom_api|127]
└─ return [Y1] → caller selects [0].tensor [LINK:custom_api:129] [CALLER:entry|49]
```

**Graph body before / after · schematic IR structure, not an exact compiler dump**

~~~text
T_cpu = native MO tensor type: float32, shape=[5,10], device=cpu:0

BEFORE custom call
builtin.module MODULE1
└─ mo.graph @addition : (T_cpu) → ()       output not finished yet
   └─ region R0 / block B0
      ├─ block argument %arg0 : T_cpu      X1 refers here
      └─ CH0: initial ordering-chain operation

AFTER insertion + device attachment + validation
builtin.module MODULE1
└─ mo.graph @addition : (T_cpu) → ()       output still not finished
   └─ region R0 / block B0
      ├─ block argument %arg0 : T_cpu
      ├─ CH0: initial ordering-chain operation
      └─ CUST1: mo.custom
         ├─ symbol attribute: "add_one"
         ├─ device attribute: kind=cpu, id=0
         ├─ operand 0: %arg0 : T_cpu
         └─ result 0: %r : T_cpu           Y1 refers here

AFTER Graph.output(Y1)                    later constructor continuation
mo.graph @addition : (T_cpu) → (T_cpu)
└─ same B0: CH0, CUST1, then mo.output(%r)
~~~

At insertion, the generic helper verifies the operation's IR structure. The call's device attachment and P1 kernel-contract verification follow in `ops.custom`. P1 remains a separate implementation package throughout; the body here contains a symbolic call, not `x.load + 1` or the Mojo source.
'''
RESULTS = '''

**Complete returned values · named wrappers and native referents**

~~~text
X1 = Python TensorValue wrapper
     native referent = block argument %arg0
     dtype = DType.float32
     shape = static [5,10]
     device = cpu:0

CUST1 = native mo.custom operation owned by G1 body B0
        symbol="add_one", operands=[%arg0], results=[%r]

Y1 = Python TensorValue wrapper
     native referent = CUST1 result %r
     dtype = DType.float32
     shape = static [5,10]
     device = cpu:0

helper return:      ([Y1], staged_op=CUST1)
ops.custom return:  [Y1]                 one-element Python list
...[0]:            Y1                   selects the first wrapper
...[0].tensor:     Y1 tensor interface   refers to the same %r
forward return:    Y1
Graph.output(Y1):  record mo.output(%r), finish output signature
~~~

The X1/Y1 labels are teaching names, not actual Python `repr` output or pointer addresses. The list contains a symbolic TensorValue, not a Buffer, NumPy array, or 50 calculated numbers. Its declared shape has 50 elements, but their values become available only through later execution. A native result is identified by its producing operation and result index; the wrapper preserves that identity.

```text
native result %r → _Value._from_cmlir(result) [LINK:ct_insert_result:1360]
└─ Value.from_mlir(...) → tensor-valued wrapper Y1 [LINK:ct_insert_result:1360] [CALLER:ct_insert_result|1360]
   ├─ helper returns results=[Y1] and staged_op=CUST1 [LINK:ct_insert_result:1364] [CALLER:ct_insert_result|1364]
   ├─ ops.custom verifies CUST1, then returns only results=[Y1] [LINK:custom_api:127] [CALLER:custom_api|129]
   └─ original lambda selects [0].tensor → constructor receives Y1 [LINK:entry:49] [CALLER:graph_init|922]
```
'''
