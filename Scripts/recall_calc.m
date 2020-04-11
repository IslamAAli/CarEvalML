x = categorical({'Naive Bayes','KNN','Decision Tree','Random Forest','SVM'});
x = reordercats(x,{'Naive Bayes','KNN','Decision Tree','Random Forest','SVM'});

dec_recall = [0.25 0.89 0.91 0.90 0.91];
bin_recall = [0.56 0.68 0.89 0.90 0.95];

y = [dec_recall; bin_recall];

figure;
b = bar(x,y);
grid on
grid minor 
xlabel('Algorithm')
ylabel('Recall')


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
