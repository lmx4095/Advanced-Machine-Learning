n = 100;
Theta = [ -10 10; -10 -10; 10 -10];
aa = [-5;+5;-5];
hidden = rand(2,n)>0.5;
cc=aa;
vv=hidden;

nvisible = size(Theta,1);
nhidden = size(Theta,2);
nsamples = size(vv,2);
assert(size(cc,1) == nvisible);
assert(size(vv,1) == nhidden);
a_b=repmat(cc,1,nsamples);
unknown=Theta*vv+a_b;
new=[];
for i=1:size(unknown,1)
    for j=1:size(unknown,2)
        new(i,j)=1./(1 + exp(-unknown(i,j)));
    end
end
