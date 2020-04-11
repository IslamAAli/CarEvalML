baseline_acc = 0.700231;

x = categorical({'Naive Bayes','KNN','Decision Tree','Random Forest','SVM'});
x = reordercats(x,{'Naive Bayes','KNN','Decision Tree','Random Forest','SVM'});

dec_acc = [0.67 0.95 0.96 0.95 0.97];
bin_acc = [0.82 0.89 0.96 0.96 0.99];
y = [dec_acc; bin_acc];

figure;
b = bar(x,y);
grid on
grid minor 
xlabel('Algorithm')
ylabel('Accuracy')


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

yline(baseline_acc)
legend ('Decimal Features', 'Binary Features', 'Baseline', 'Location','northwest')
