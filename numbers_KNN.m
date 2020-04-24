nr_test = 10000; 
nr_train = 60000;
k= 7; 

guess = zeros(1, nr_test); %guesses for the test images

fprintf('Calculates the 7-NN with %d training images, and %d test images\n', nr_train, nr_test)

for i = 1:nr_test
    test_inst = testv(i,:);
    Eu_dist = dist(trainv(1:nr_train,:), test_inst'); %finds the euclidian distance
    [d,index] = mink(Eu_dist,k); %picks the 7 smallest distances 
    pred = trainlab(index); %Finds the label for the closes images  
    
    guess(i) = mode(pred); %Guesses on the most common predinction
end

disp("Confusion matrix and correctness:")

%Makes the confusion matrix and correctness, and plots them
conf = zeros(10,10);
for i = 1:nr_test
    conf((guess(i)+1),(testlab(i)+1))= conf((guess(i)+1),(testlab(i)+1)) + 1;
end

T = array2table(conf, 'VariableNames', {'zero', 'one', 'two','three','four', 'five','six','seven','eight','nine'}, 'RowNames',{'zero', 'one', 'two','three','four', 'five','six','seven','eight','nine'} );
correct = trace(conf)/nr_test;

disp(T)
fprintf('Correct: %0.3f\n\n', correct)