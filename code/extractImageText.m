function [text] = extractImageText(fname)
% [text] = extractImageText(fname) loads the image specified by the path 'fname'
% and returns the next contained in the image as a string.

    %Loading the image
    im = imread(fname);

    %Finding the letters in the image
    [lines, bw] = findLetters(im);
    %bw = bwmorph(bw,'thicken');
    
    num_lines = length(lines);

    %Loading the Trained Model Weights and bias
    load('../../../best_fine_tuned.mat');
    W = best_w;
    b = best_b;
    
    %Vocab
    vocab= ['A','B','C','D','E','F','G','H','I','J',...
        'K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9'];
    
    %Final text list initialization
    final ={};
    figure;
    imshow(im);
    hold on ;
    %Traversing through each line in the image
    for num = 1:num_lines
        %Traversing through each character in the line
        test_line = [];
        
        %Checking for spaces
        spaces_diff = lines{num}(1:end-1,3) -lines{num}(2:end,1);
        
        space_indexes = find(abs(spaces_diff)>1.5*mean(abs(spaces_diff)));
        
        for character = 1: size(lines{num},1)
            
            rectangle('Position',[lines{num}(character,1)-16,lines{num}(character,2)-16,...
                lines{num}(character,3)-lines{num}(character,1)+32,...
                lines{num}(character,4)-lines{num}(character,2)+32],...
        'EdgeColor','r','LineWidth',2 );
            im = bw(round(lines{num}(character,2)):...
                ceil(lines{num}(character,4)),...
                round(lines{num}(character,1)):...
                ceil(lines{num}(character,3)));
            
            %Zero Padding for the character image
            im = padarray(im,[16 16],0,'both');
            
            %Resizing to 32x32
            im = imresize(~im,[32,32]);
            test_line(character,:) = im(:);
            
        end
       
        [outputs] = Classify(W,b,test_line);
        [~,ind] = max(outputs');
     
        voc = vocab(ind);
        for n = 1:length(space_indexes)
            voc = insertAfter(voc,space_indexes(n),' ');
            space_indexes = space_indexes+1;
        end
        voc = strcat(voc, newline);
        final{num} = voc;       
    end
    text = final'
end