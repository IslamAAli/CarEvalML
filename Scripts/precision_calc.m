x = categorical({'Naive Bayes','KNN','Decision Tree','Random Forest','SVM'});
x = reordercats(x,{'Naive Bayes','KNN','Decision Tree','Random Forest','SVM'});

dec_precision = [0.69 0.89 0.88 0.83 0.88];
bin_precision = [0.77 0.82 0.85 0.83 0.92];

y = [dec_precision; bin_precision];

figure;
b = bar(x,y);
grid on
grid minor 
xlabel('Algorithm')
ylabel('Precision')


xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints*0+0.2;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1 = b(2).XEndPoints;
ytips1 = b(2).YEndPoints*0+0.2;
labels1 = string(b(2).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

legend ('Decimal Features', 'Binary Features', 'Location','northwest')
