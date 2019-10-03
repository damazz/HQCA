OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
x q[0];
h q[0];
x q[1];
h q[1];
cx q[0],q[1];
h q[2];
cx q[1],q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
u1(1) q[3];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[1];
h q[2];
h q[2];
z q[2];
h q[3];
s q[3];
h q[3];
h q[4];
sdg q[5];
h q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[2],q[3];
u1(0.500000000000000) q[3];
h q[3];
cx q[3],q[2];
sdg q[2];
h q[2];
s q[2];
z q[4];
cx q[3],q[4];
z q[3];
h q[3];
u1(0.500000000000000) q[3];
cx q[2],q[3];
h q[2];
s q[2];
sdg q[4];
h q[4];
s q[4];
cx q[4],q[3];
h q[3];
cx q[5],q[4];
h q[4];
s q[4];
h q[5];
s q[5];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];