cdir := "/home/scott/Documents/research/software/hqca/hqca/maple/test_cases/":
with(LinearAlgebra): with(ArrayTools): with(QuantumChemistry):
loaddata := readdata(cat(cdir,"tmp7bh4a4ao.rdm"), 8):
Flatten := proc(x) local n, a, i, j, k, l; `local`(a, n, i, j, k, l); `description`("convert chemists to numpy and flatten an array to form a matrix"); n := Size(x); a := Matrix(1 .. n[1]*n[2], 1 .. n[3]*n[4], datatype = float[8]); for i to round(n[1]) do for j to round(n[2]) do for k to round(n[3]) do for l to round(n[4]) do a[(i - 1)*n[1] + j, (k - 1)*n[3] + l] := x[i, j, k, l]; end do; end do; end do; end do; return a; end proc:
New := Array(1 .. 4, 1 .. 4, 1 .. 4, 1 .. 4, datatype = float[8]):
for i in loaddata[3 .. ()] do
    New[round(i[1]), round(i[2]), round(i[3]), round(i[4])] := i[5]:
end do:
Transpose(Re(Eigenvalues(Flatten(New))));
pure := Purify2RDM(New, spin_free = false, electron_number=4, conv_tol = 0.00000000001):
ExportMatrix(cat(cdir, "_temp_purified.csv"), Flatten(pure[rdm2])):
