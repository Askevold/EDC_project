%Før man kan kjøre koden må man ha kjørt read09.m for å hente ut data. 
nr_test = 100; 
nr_train = 1000;

guess = zeros(1, nr_test); %guesses for the test images

fprintf('Calculates the NN with %d training images, and %d test images\n', nr_train, nr_test)

for i = 1:nr_test
    test_inst = testv(i,:);
    Eu_dist = dist(trainv(1:nr_train,:), test_inst'); %finds the euclidian distance
    [d,index] = min(Eu_dist); %picks the smallest distance 
    pred = trainlab(index); %Finds the label for the closes image  
    
    guess(i) = pred;
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

disp("Plotting an incorrect and a correct numbers")
In_idx = []; %list of indexes for incorrect guesses
Co_idx = []; %list of indexes for correct guesses
for i = 1:nr_test
   if guess(i) ~= testlab(i) 
       In_idx = [In_idx,i];
   else
       Co_idx = [Co_idx,i];
   end
end

rand_in_idx = randi(length(In_idx)); %Pulls a random index from the incorrect guesses
%Makes the random index into a printable image, then flips and rotates so it's correct 
x = zeros(28,28); 
x(:)= testv(In_idx(rand_in_idx),:);
in_img = flip(imrotate(x,-90), 2); 

rand_cor_idx = randi(length(Co_idx)); %Pulls a random index from the correct guesses
%Makes the random index into a printable image, then flips and rotates so it's correct
x = zeros(28,28); 
x(:)= testv(Co_idx(rand_cor_idx),:);
co_img = flip(imrotate(x,-90), 2);

%plots the incorrect and correct images 
subplot(1,2,1), imshow(in_img), title(sprintf('%d incorrectly classyfied as %d', testlab(In_idx(rand_in_idx)),guess(In_idx(rand_in_idx)) ) ) 
subplot(1,2,2), imshow(co_img), title(sprintf('Correctly classyfied %d', testlab(Co_idx(rand_cor_idx))) ) 
    





