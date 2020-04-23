M = 64;

%sorts the train matrix from 0 to 9
[sort_trainlab, sort_index] = sort(trainlab);
sort_trainv = zeros(num_train,n);
for i = 1:num_train
    sort_trainv(i,:) = trainv(sort_index(i),:);
end

%clustering of all the classes
trainv_C = zeros(M*10,n);
first = 1;
for i = 0:9
     last = find (sort_trainlab == i,1, 'last'); %find last index of the class 
     class_i = sort_trainv(first : last,:);
     
     [idxi, Ci] = kmeans(class_i, M); % clusters the class into 64 templates
     fprintf('Done clustering class %d\n',i)
     trainv_C((i*M+1):((i+1)*M),:) = Ci; 
     
     first = last;
end

%NN classification
nr_test = 1000; 

guess = zeros(1, nr_test); %guesses for the test images

fprintf('Calculates the NN with %d training images, and %d test images\n', nr_train, nr_test)

for i = 1:nr_test
    test_inst = testv(i,:);
    Eu_dist = dist(trainv_C, test_inst'); %finds the euclidian distance
    [d,index] = min(Eu_dist); %picks the smallest distance 
    pred = testlab(index); %Finds the label for the closes image  
    
    guess(i) = pred;
end

%Plots the confusion matrix and correctness
conf = zeros(10,10);
for i = 1:nr_test
    conf((guess(i)+1),(testlab(i)+1))= conf((guess(i)+1),(testlab(i)+1)) + 1;
end

T = array2table(conf, 'VariableNames', {'zero', 'one', 'two','three','four', 'five','six','seven','eight','nine'}, 'RowNames',{'zero', 'one', 'two','three','four', 'five','six','seven','eight','nine'} );
correct = trace(conf)/nr_test;

disp(T)
fprintf('Correct: %0.3f\n\n', correct)
