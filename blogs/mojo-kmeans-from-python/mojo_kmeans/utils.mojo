from .matrix import Matrix
fn list_to_matrix[dtype: DType](lst: List[Matrix[dtype]]) -> Matrix[dtype]:
    var new_mat = Matrix[dtype](len(lst),len(lst[0]))
    var tmpPtr = new_mat._matPtr
    for arr in lst:
        memcpy(tmpPtr, arr[]._matPtr, len(arr[]))
        tmpPtr += len(arr[])
    return new_mat