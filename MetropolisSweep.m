function MA=MetropolisSweep(theta, beta)
[Nt,Nx,~]= size(theta); %we're ignoring the third output Nd=2
    for t=1:Nt
       for x=1:Nx
           for mu = 1:2
               nu=3-mu;
               Otheta= theta(t,x,mu);
               % random should be a random phase symetrically distributed around 0
               %so r=mean+std*randn
               Ntheta=Otheta+randn(1);
               % Compute the difference in action
               % Do not compute the whole action, as that is inefficient!
               U_old=exp(1i*Otheta);
               U_new=exp(1i*Ntheta);
               % We can instead use the staple/Eq. 102 here
               A_field=Staple(theta,mu,nu);
               A_local=A_field(t,x);
               DeltaS=-beta*real((U_new-U_old)*conj(A_local));
               p=exp(-DeltaS);
               % Accept-reject step
               q = rand(1); % Random number uniformly distributed between 0 and 1
               if q <= p
                   % Accept the proposed update
                   theta(t,x,mu) = Ntheta; %update the specific link
               end
            end
        end
    end
    MA=theta; %updated lattice configurations
end