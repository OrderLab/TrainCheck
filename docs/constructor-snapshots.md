# Constructor PRE snapshots

When argument dumping is enabled for a class's `__init__`, TrainCheck records
the receiver before the initializer body runs. The receiver may be empty,
partially initialized by a subclass, or being initialized again. The instrumenter
identifies initializers from their owning class and attribute name; a standalone
function named `__init__` keeps ordinary argument serialization.

The receiver is encoded under its static, fully qualified type name with an
`_TRAINCHECK_CONSTRUCTOR_SNAPSHOT` marker. This is a deliberate change from
ordinary object introspection for **all selected initializers**, including
metaclass initializers. Other arguments and ordinary method calls keep their
existing representation. The function's PRE, POST, and exception records remain.

## Stored-state contract

The marker contains:

| Field | Meaning |
| --- | --- |
| `phase` | `before_init`; this does not claim initialization succeeded. |
| `coverage` | `stored_attributes_only`; computed properties are not captured. |
| `object_id` | Identity of the actual receiver. |
| `dictionary_status` | `captured`, `absent`, or `unavailable` when storage exists but no native accessor remains accessible. |
| `attributes` | Selected entries from the instance dictionary or native class namespace. |
| `slots` | Selected populated native slots, each with its declaring `owner_type`, `name`, and tagged `value`. |
| `unset_slots` | Selected declared slots whose storage has not been assigned. |
| `requested_attributes` | The explicit selector, or `null` for all accessible stored fields. |
| `unavailable_attributes` | Requested names with no captured dictionary entry or populated slot. |

Dictionary entries and slot records are separate. A base slot, a derived slot,
and a dictionary entry can all have the same name and hold different values;
none overwrites another. Slot selectors use the stored name, including Python's
name mangling for private slots. A requested name selects each matching storage
location. An unset location stays explicit even if another matching slot has a
value. An empty selector is distinct from unselected/all-field collection.

Exact built-in scalars retain their values. Nonfinite floats have explicit
`nan`, `+inf`, and `-inf` tags. Other values, including containers and tensors,
are **identity/type references**, not recursive content snapshots or tensor
hashes. Missing storage is never represented as a stored `None` value.

## Avoiding application execution

The snapshot reads native dictionary and slot descriptors from raw class
namespaces. It bypasses receiver `__getattribute__`, `__getattr__`, `__dir__`,
properties, custom descriptors, and custom metaclass lookup/properties. It also
bypasses overridden dictionary iteration methods and custom metaclass equality.
Constructor receivers are excluded from proxy-marker scans and argument
unproxying on this dumping path; ordinary arguments keep those behaviors.

Normal positional construction and explicit keyword-receiver calls are covered.
If a native callable exposes no signature, its positional receiver is supported;
no keyword receiver name is guessed. Missing-receiver errors are produced by
the original callable and retain an exception record.

Native extension state without a recognized storage descriptor, replaced slot
descriptors, and computed attributes are outside this snapshot's coverage.
Dictionary storage hidden behind a replacement property is marked unavailable
unless an inherited native accessor still exposes it. Snapshots are synchronous,
do not cache application metadata, and do not promise an atomic view during
concurrent mutation of the same object's storage.

Existing invariants that depended on introspected constructor receiver fields
need recollection and activation validation against this representation. A
requested property is explicitly unavailable, not a successfully observed value.
