%%
%ZOH när A^2 inte är 0 med invLaplace
%clc
%clear all
syms s h a b c
A=[a 0; b c];
B=[0; d];
C=[1 0];

x=inv([s 0; 0 s]-A)
Phi=ilaplace(x, s)

Gamma=int(Phi*B, s, 0, h)