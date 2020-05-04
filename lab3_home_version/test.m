clear all
clc

%%This file is used only for calculating appropriate variables, for example
%%when doing pole placements etc.
%% Assignment 1
s1 = -3+2j; %Pole 1
s2 = -3-2j; %Pole 2
syms K Ti %Declaring variables to find solutions for
E = [Ti*(s1^2 + s1*(0.12 + 2.25*K))+2.25*K == 0, Ti*(s2^2 + s2*(0.12 + 2.25*K))+2.25*K == 0]; 
%Two equations, two unknowns.
%These equations were retrieved by calculating the closed looped system and
%setting the denominator to 0
S = solve(E,K,Ti) %Solving equations E for Ti and K

K = S.K(2) %Resulting K for the solver
Ti = S.Ti(2) %Resulting Ti for the solver

%% Assignment 5
%Matrix declarations
A = [-0.12 0; 5 0]; 
B = [2.25; 0];
C = [0 1];
s = [0.8+0.1j 0.8-0.1j]; %Poles

sys = ss(A,B,C,0); %From defined matrices
[SYSD, G] = c2d(sys, 0.05);
[L,prec,message] = place(SYSD.A,SYSD.B,s);

acl = A-B*L;
sys2 = ss(acl, B, C, 0);
SYSD5 = c2d(sys2, 0.05);
lr = 1/dcgain(SYSD5);

%% Assignment 6
A = [-0.12 0; 5 0];
B = [2.25; 0];
C = [0 1];
s6 = [0.6-0.2*j 0.6+0.2*j 0.55];

phi_e = [SYSD.A SYSD.B; 0 0 1];
gamma_e = [SYSD.B; 0];
C_e = [SYSD.C 0];

K = place(phi_e',C_e',s6)


