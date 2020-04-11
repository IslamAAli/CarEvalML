X = categorical({'unacc','acc','good','v-good'});
X = reordercats(X,{'unacc','acc','good','v-good'});
Y = [1210 384 69 65];
b = bar(X,Y, 0.4);
grid on
grid minor 
xlabel('Car Classes')
ylabel('Frequency')
legend Frequency
ylabel('Data Samples')

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')