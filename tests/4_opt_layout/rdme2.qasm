OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
s q[2];
z q[2];
z q[3];
cx q[0],q[2];
cx q[1],q[0];
cx q[3],q[1];
h q[3];
rz(pi/8) q[3];
cx q[2],q[3];
rz(-pi/8) q[3];
cx q[1],q[3];
rz(pi/8) q[3];
cx q[2],q[3];
rz(-pi/8) q[3];
h q[3];
cx q[3],q[1];
cx q[1],q[0];
cx q[0],q[2];
cz q[3],q[1];
s q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
