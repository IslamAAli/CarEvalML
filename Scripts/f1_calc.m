x = categorical({'Naive Bayes','KNN','Decision Tree','Random Forest','SVM'});
x = reordercats(x,{'Naive Bayes','KNN','Decision Tree','Random Forest','SVM'});

dec_f1 = [0.21 0.89 0.88 0.85 0.88];
bin_f1 = [0.62 0.73 0.86 0.86 0.94];

y = [dec_f1; bin_f1];

figure;
b = bar(x,y);
grid on
grid minor 
xlabel('Algorithm')
ylabel('F1-Measure')


xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints*0+0.1;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

xtips1 = b(2).XEndPoints;
ytips1 = b(2).YEndPoints*0+0.1;
labels1 = string(b(2).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

legend ('Decimal Features', 'Binary Features', 'Location','northwest')
