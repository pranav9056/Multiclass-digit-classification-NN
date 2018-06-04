function [a3,a2,z3,z2] = feedForward(Theta1,Theta2,X)
    m = size(X, 1);
    theta1 = Theta1';
    z2  = X*theta1;
    a2 = sigmoid(z2);
    a2 = [ones(m, 1) a2];
    theta2 = Theta2';
    z3 = a2 * theta2;
    a3 = sigmoid(z3);

end