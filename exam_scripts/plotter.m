% Edvin 23/4 2020
clc
clear all
close all
phi = [1 1 0.5; 0 1 1; 0 0 1];
gamma = [1/6; 1/2; 1];
C = [1 0 0];
D = 0;
h = 1;
sys = ss(phi, gamma, C, D, h);
H = tf(sys);
zerop = zpk(H)
%% Bode
figure
bode(H)
%% Step Response
figure
step(H)
%% Nyquist
figure
nyquist(H)