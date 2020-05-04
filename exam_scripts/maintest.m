%% PQ - formula
syms s
eq = s^2 + 6*s + 9 == 0;
ans = solve(eq,s)


%% ZOH/INVERS/LAPLACE/TRANSFERFUNCTION
syms s a b c d

A = [ a 0;
      b c];
answer = A*A

sI = [s 0;
      0 s];
sI_A = sI - A

Ainv = inv(sI_A)


C = [1 0];
B = [0; d]

G = C*Ainv*B
%% Characteristic polynom - L
sympref('FloatingPointOutput',true)
syms z l1 l2 h

zI = [z 0; 0 z]
Phi = [1 0; h 1]
Gamma = [2*h; h^2]
L = [l1 l2]

preStep = zI - Phi + Gamma*L
poly = det(zI - Phi + Gamma*L)

c = coeffs(poly, z)


%% Solve equation systems
syms l1 l2
sympref('FloatingPointOutput',true)

eq1 = c(2) == 0     %One z
eq2 = c(1) == 0     %No z

eqns = [eq1, eq2]

S = solve(eqns, [l1 l2])

S.l1
S.l2

%% Characteristic polynom - K
sympref('FloatingPointOutput',true)
syms z k1 k2 h
zI = [z 0; 0 z]
Phi = [0.5 0.25; 0.5 0.5]
Gamma = [1; 0]
K = [k1; k2];
C = [0 0.25];

hello = Phi - K*C
preStep = zI - Phi + K*C
poly = det(preStep)

c = coeffs(poly, z)


%% Solve equation systems
syms k1 k2
sympref('FloatingPointOutput',true)

eq1 = c(2) == -0.5 %One Z
eq2 = c(1) == 0 %No Z

eqns = [eq1, eq2]

S = solve(eqns, [k1 k2])

double(S.k1)
double(S.k2)


%% Static gain
I = [1 0; 0 1];
L = [ 1, 0]
lr = 1/(C*inv(I - Phi + Gamma*L)*Gamma)

%% Pulse transfer functions
syms z

HC1 = (z - 0.75)/(z-0.15)

HC2 = (10*(z-0.75)^2)

HP = (0.01*(z+0.95))/(z-0.95)^2

HCL = (HC1*HP)/(1+HC1*HP)


answer = simplify(HCL)

%% Discretize s simply
syms K Td N y q
s=(2*(q-1))/(h*(q+1))
D = (-K*s*Td*y)/(1+(s*Td)/N)