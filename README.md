Первая лабораторка по надежности инф. систем
<h5>Требования к входным данным(in.txt):</h5>
В первой строке: придел времени и шаг изменения времени, в остальных сроках обьекты иследования: закон распределения(цифрой) 
и их параметры через пробел. Каждый объект с новой строки.<br>
<h6>Допустимые значения распределений:</h6>
<ol>
<li>
 Экспоненциальное распределение
</li>
<li>
 Равномерное распределение
</li>
<li>
 Гамма распределение
</li>
<li>
 Усеченное нормальное распределение
</li>
<li>
 Распределение Рэлея
</li>
<li>
 Распределение Вейбулла
</li>
<li>
 Нормальное распределение
</li>
</ol>
<h5>Requirements:</h5>
<ul>
<li>
python 3.x
</li>
<li>
numpy
</li>
<li>
scipy
</li>
<li>
matplot
</li>
</ul>
<h5>Для пользователей Windows если чтото пошло не так с запуском:</h5>
<ol>
<li>
Установить Vagrant, Git
</li>
<li>
Создать рабочую папку для Vagrant и поместить туда Vagrantfile
</li>
<li>
Изменить в Vagrantfile "C:\\important\\NIS" на папку в которую вы скачали проект
</li>
<li>
Запустить Git Bash
</li>
<li>
Ввести команды <code>vagrant up</code>, <code>vagrant ssh</code> 
</li>
<li>
Ввести команды <code>cd NIS/1</code>, <code>python lab1.py</code> 
</li>
</ul>