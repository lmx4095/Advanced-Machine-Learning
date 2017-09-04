nv = 3; nh = 2;
n = 100;
% ground truth params
Theta = [ -10 10; -10 -10; 10 -10];
bb = [2;2]; aa = [-5;+5;-5];
hidden = rand(nh,n)>0.5;
% sample from p(h,v)
for it=1:100
    visible = sample(Theta,aa,hidden);
    hidden = sample(Theta',bb,visible);
end
% step size
eta = 0.05;
% momentum
mom = 0.95;
% learned parameters
lTheta = 0.1*randn(size(Theta));laa = zeros(size(aa));lbb = zeros(size(bb));
% update direction
vt = zeros(size(Theta));vaa = zeros(size(aa));vbb = zeros(size(bb));
for it=1:10000
    [gt,ga,gb,recon] = cdgradient(lTheta,laa,lbb,visible);
    vaa = mom*vaa + eta*ga; vbb = mom*vbb + eta*gb; vt = mom*vt + eta*gt;
    lTheta = lTheta + vt; laa = laa + vaa; lbb = lbb + vbb;
    if (mod(it,100) == 0)
        fprintf('Iter: %d recon: %g ',it,recon);
        fprintf('Distance of learned theta to ground truth theta: %g\n',...
            min([norm(lTheta - Theta) norm(lTheta - Theta(:,[2 1]))]))
    end
end
