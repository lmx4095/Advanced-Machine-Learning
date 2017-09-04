function [bg_theta,bg_aa,bg_bb,recon] = cdgradient(Theta,aa,bb,V)
p = size(V,1);
T = size(V,2);
nhidden = size(Theta,2);
nvisible = size(Theta,1);
assert(length(aa) == nvisible)
assert(length(bb) == nhidden)
bg0_theta = zeros(size(Theta));
bg1_theta = bg0_theta;
bg0_aa = zeros(size(aa));
bg1_aa = bg0_aa;
bg0_bb = zeros(size(bb));
bg1_bb = bg0_bb;
recon = 0;
for t=1:T
    vt = V(:,t);
    ht = sample(Theta',bb,vt);
    bg0_theta = bg0_theta + vt*sigmoid(bb+Theta'*vt)'/T;;
    bg0_aa = bg0_aa + vt/T;
    bg0_bb = bg0_bb + sigmoid(bb+Theta'*vt)/T;
    vt1 = sample(Theta,aa,ht);
    ht1 = sample(Theta',bb,vt1);
    bg1_theta = bg1_theta + vt1*ht1'/T;
    bg1_aa = bg1_aa + vt1/T;
    bg1_bb = bg1_bb + ht1/T;
    recon = recon + norm(vt1 - vt);
end
bg_theta = bg0_theta - bg1_theta;
bg_aa = bg0_aa - bg1_aa;
bg_bb = bg0_bb - bg1_bb;