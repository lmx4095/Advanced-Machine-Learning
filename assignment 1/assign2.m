nv = 3;
nh = 2;
n = 100;
Theta = [ -10 10; -10 -10; 10 -10];
bb = [2;2];
aa = [-5;+5;-5];
hidden = rand(2,n)>0.5;
for it=1:1000
    visible = sample(Theta,aa,hidden);
    newhidden = sample(Theta',bb,visible);
    err = (sum(sum(newhidden ~= hidden))/prod(size(hidden)))
    hidden = newhidden;
end
hist([1 2]*hidden,[0:3])