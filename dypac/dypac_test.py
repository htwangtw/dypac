
n_batch = 2
part_batch = np.array([0, 1, 1, 2, 2, 0, 0, 0, 1, 2])
part_cons = np.array([2, 1, 0, 2, 0, 1])
index_batch = [0, 5, 10]
index_cons = [0, 3, 6]
part = _propagate_part(part_batch, part_cons, n_batch, index_batch, index_cons)
# part = np.array([2, 1, 1, 0, 0, 2, 2, 2, 0, 1])