function P = PlaquetteTest(theta,T,X)
    thetaT=theta(:,:,T);
    thetaX=theta(:,:,X);
    shiftT=[0,0];
    shiftT(T)=-1;%eg [-1,0] if T is 1
    shiftX=[0,0];
    shiftX(X)=-1;%if X=2 then [0,-1]
    theta_X_shift_T=circshift(thetaX,shiftT); %evaluating at n+T and n+X
    theta_T_shift_X=circshift(thetaT,shiftX);
    arg=thetaT+theta_X_shift_T-theta_T_shift_X-thetaX;
    P=exp(1i*arg);
end