
with(LinearAlgebra);
cdir := "/home/scott/wDocuments/research/software/hqca/hqca/maple/";
data := readdata(cat(cdir, "temp.rdm"), 6);
with(ArrayTools);
# 
Rearrange := proc(x) local n, a, i, j, k, l; `local`(a, n, i, j, k, l); `description`("convert chemists to numpy and flatten an array to form a matrix"); n := Size(x); a := Array(1 .. n[1], 1 .. n[2], 1 .. n[3], 1 .. n[4], datatype = float[8]); for i to n[1] do for j to n[2] do for k to n[3] do for l to n[4] do a[i, k, j, l] := x[i, j, k, l]; end do; end do; end do; end do; return a; end proc;
Flatten := proc(x) local n, a, i, j, k, l; `local`(a, n, i, j, k, l); `description`("convert chemists to numpy and flatten an array to form a matrix"); n := Size(x); a := Array(1 .. n[1]*n[2], 1 .. n[3]*n[4], datatype = float[8]); for i to n[1] do for j to n[2] do for k to n[3] do for l to n[4] do a[(i - 1)*n[1] + j, (k - 1)*n[3] + l] := x[i, k, j, l]; end do; end do; end do; end do; return a; end proc;
New := Array(1 .. 3, 1 .. 3, 1 .. 3, 1 .. 3, datatype = float[8]);
newdata := data[3 .. ()];
for i in newdata do
    New[round(i[1]), round(i[2]), round(i[4]), round(i[3])] := i[5];
end do;
New[1, 2];
with(QuantumChemistry);

mol := [["H", 1.00000000, 0, 0], ["H", -1.00000000, 0, 0]];

h2 := Variational2RDM(mol, basis = "STO-3G", return_rdm = "rdm1_and_rdm2");



h2[rdm2];

pure := Purify2RDM(h2[rdm2], spin_free = true, electron_number = 2, conv_tol = 0.10000000, conditions = "DQG");
^2 D ^{i,j}_{k,l} = < i j l k  > -> A[1..r,1..r,1..r,1..r] <=> A[i,j,k,l]  
;

G := Flatten(h2[rdm2]);

Eigenvalues(G);
F := Flatten(pure[rdm2]);


Eigenvalues(F);


