function A = Staple(theta,mu,nu)
    theta_mu= theta(:,:,mu);
    theta_nu=theta(:,:,nu);

    shift_mu=[0,0]; shift_mu(mu) = -1;
    shift_nu=[0,0]; shift_nu(nu) = -1;
    shift_minus_nu=[0,0]; shift_minus_nu(nu) = 1;

    %forward Staple
    theta_mu_shift_nu=circshift(theta_mu, shift_nu);
    theta_nu_shift_mu = circshift(theta_nu, shift_mu);
    argf= theta_nu + theta_mu_shift_nu - theta_nu_shift_mu;
    FA=exp(1i*argf);

    %backward staple 
    theta_nu_minus_nu = circshift(theta_nu, shift_minus_nu);
    theta_mu_minus_nu = circshift(theta_mu, shift_minus_nu);
    theta_nu_shift_mu_minus_nu = circshift(theta_nu_shift_mu, shift_minus_nu);
    
    argb= -theta_nu_minus_nu + theta_mu_minus_nu + theta_nu_shift_mu_minus_nu;
    BA=exp(1i * argb);

    %returning the full Nt by Nx matrix of staples
    A= FA + BA;
end