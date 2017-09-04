function h = sample(Theta,cc,vv)
nvisible = size(Theta,1);
nhidden = size(Theta,2);
nsamples = size(vv,2);
assert(size(cc,1) == nvisible);
assert(size(vv,1) == nhidden);
a_b=repmat(cc,1,nsamples);
unknown=Theta*vv+a_b;
for i=1:size(unknown,1)
    for j=1:size(unknown,2)
        h(i,j)=sigmoid(unknown(i,j));
    end
end