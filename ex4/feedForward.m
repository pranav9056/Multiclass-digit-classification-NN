function a2 = feedForward(Theta1,Theta2,X)
    m = size(X, 1);
    theta1 = Theta1';
    z1  = X*theta1;
    a1 = sigmoid(z1);
    a1 = [ones(m, 1) a1];
    theta2 = Theta2';
    z2 = a1 * theta2;
    a2 = sigmoid(z2);

end