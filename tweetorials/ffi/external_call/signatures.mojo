fn external_call[
    callee: StringLiteral, return_type: AnyRegType
]() -> return_type:
    ...

fn external_call[
    callee: StringLiteral, return_type: AnyRegType, T0: AnyRegType
](arg0: T0) -> return_type:
    ...

fn external_call[
    callee: StringLiteral,
    return_type: AnyRegType,
    T0: AnyRegType,
    T1: AnyRegType,
](arg0: T0, arg1: T1) -> return_type:
    ...

