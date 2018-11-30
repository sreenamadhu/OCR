function [lines, bw] = findLetters(im)
% [lines, BW] = findLetters(im) processes the input RGB image and returns a cell
% array 'lines' of located characters in the image, as well as a binary
% representation of the input image. The cell array 'lines' should contain one
% matrix entry for each line of text that appears in the image. Each matrix entry
% should have size Lx4, where L represents the number of letters in that line.
% Each row of the matrix should contain 4 numbers [x1, y1, x2, y2] representing
% the top-left and bottom-right position of each box. The boxes in one line should
% be sorted by x1 value.
    
    close all;
    
    %Preprocessing...
    gray_im = rgb2gray(im);
    bin_im = imbinarize(gray_im);

    %Removing Noise 
    bw = medfilt2(~bin_im);
    bw = bwareaopen(bw,30);
    bw = bwmorph(bw,'thicken');
    bw = double(bw);
    bw = imgaussfilt(bw,1.2);
    
    %Finding the connected points 
    connectivity = bwconncomp(bw);
    l = labelmatrix(connectivity);
    
    %Extracting the bounding boxes around the characters
    bb = regionprops(l,'BoundingBox');
    bb = cell2mat((struct2cell(bb))');
    
    %Sorting the boundary boxes..so the boxes are ordered left to right
    [val,ind]= sort(bb(:,2));
    bb = bb(ind,:);
    
    %Ignoring the small characters like dots,etc
    bb_area = regionprops(l,'Area');
    bb_area = cell2mat((struct2cell(bb_area(ind,:)))');
%    remove_indexes = find(bb_area<min(bb_area)+100 & bb_area>min(bb_area)-100); 
    remove_indexes = find(bb_area<500);
    
    %Finding the character lines in the images
    hori = [bb(2:length(bb),1);0]-bb(1:length(bb),1);
    vert = [bb(2:length(bb),2);0]-bb(1:length(bb),2);
    dimen = [hori,vert];
   
   %Finding the boundary boxes for which vertical change is huge and there
   %exists a minimum horizontal difference
   index = find(dimen(:,2)>mean(dimen(1:end-1,2)) & abs(dimen(:,1))>min(abs(dimen(:,1))));
   %Ignoring small changes if those boundary boxes sizes in the line varies
   lines_indexes = [0,index(find(dimen(index,2)>bb(index,4)))',length(bb)];
 %  lines_indexes(ismember(lines_indexes,remove_indexes))=remove_indexes - 1;
   lines_indexes(ismember(lines_indexes,remove_indexes))=lines_indexes(ismember(lines_indexes,remove_indexes)) - 1;
   lines_indexes = unique(lines_indexes);
  
   for i = 1:length(lines_indexes)-1
       line = [];
        %figure;
        %imshow(bw);
        %hold on;
        %[:,:,col,row] - boundary 
        for num = lines_indexes(i)+1: lines_indexes(i+1)
       %     rectangle('Position',[bb(num,1),bb(num,2),bb(num,3),bb(num,4)],...
        %'EdgeColor','r','LineWidth',2 )            
            if ~any(remove_indexes==num)
                line = [line;[bb(num,1),bb(num,2),bb(num,1)+bb(num,3),bb(num,2)+bb(num,4)]];
            end

        end
        
        [~,ind]=sort(line(:,1));
        lines{i} = line(ind,:);
    end
    assert(size(lines{1},2) == 4,'each matrix entry should have size Lx4');
    assert(size(lines{end},2) == 4,'each matrix entry should have size Lx4');
    lineSortcheck = lines{1};
    assert(issorted(lineSortcheck(:,1)) | issorted(lineSortcheck(end:-1:1,1)),'Matrix should be sorted in x1');
end
