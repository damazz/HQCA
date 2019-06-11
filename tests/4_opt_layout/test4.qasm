OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
x q[0];
x q[1];
u3(-1.57079632679490,-pi/2,pi/2) q[0];
h q[1];
cx q[0],q[1];
h q[2];
cx q[1],q[2];
h q[3];
cx q[2],q[3];
u1(pi/2) q[3];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
u3(1.57079632679490,-pi/2,pi/2) q[0];
h q[1];
h q[2];
h q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[3],q[4];
h q[4];
cx q[3],q[2];
cx q[2],q[3];
h q[3];
cx q[2],q[1];
cx q[1],q[2];
h q[2];
cx q[1],q[0];
cx q[0],q[1];
h q[1];
measure q[0] -> c[4];
measure q[1] -> c[0];
measure q[2] -> c[1];
measure q[3] -> c[2];
measure q[4] -> c[3];