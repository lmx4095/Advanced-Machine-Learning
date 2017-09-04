img = loadMNISTImages('train-images-idx3-ubyte');
lab = loadMNISTLabels('train-labels-idx1-ubyte');
nv = size(img,1);
nh = 100;
n = size(img,2);
% step size
eta = 0.001;
% momentum
mom = 0.8;
% learned parameters
lTheta = 0.1*randn(nv,nh);laa = zeros(nv,1);lbb = zeros(nh,1);
% update direction
vt = zeros(size(lTheta));vaa = zeros(size(laa));vbb = zeros(size(lbb));
minibatch = 200;
last = 0;
list = randperm(n);
ct = 0;
ITER = 3000;
d = sqrt(nv);
f = sqrt(nh);
skip = 10;
makemovie = 1;
if makemovie
    frames = zeros((d+1)*f+1,(d+1)*f+1,1,ceil(ITER/skip));
end
for it=1:ITER
    idxs = list(mod(last:last+minibatch-1,n)+1);
    last = last+minibatch;
    visible = img(:,idxs);
    eta = eta*0.9999;
    [gt,ga,gb,recon] = cdgradient(lTheta,laa,lbb,visible);
    vt = mom*vt + eta*gt;vaa = mom*vaa + eta*ga;vbb = mom*vbb + eta*gb;
    lTheta = lTheta + vt; laa = laa + vaa; lbb = lbb + vbb;
    if (mod(it,skip) == 0)
        fprintf('Iter: %d Recon: %d\n',it,recon);
        if makemovie
            ct = ct+1;
            frames(:,:,1,ct) = showfilters(lTheta);
        end
    end
end
if makemovie
    frames = frames(:,:,:,1:ct);
    frames = frames - min(frames(:));
    frames = frames./max(frames(:));
    frames = uint8(frames*255);
    mov = immovie(frames,gray(256));
    writerObj = VideoWriter('learning.mpg','MPEG-4');
    open(writerObj);
    writeVideo(writerObj,mov);
    close(writerObj);
end