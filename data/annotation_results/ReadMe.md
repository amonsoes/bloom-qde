# Annotations Results

## Majority Vote
Die Dateien bei denen die Klassen nach Mehrheitsstimme bestimmt wurden haben folgenden Namenschema:
`annotation_results_{str_choice}_choice_th{THRESHOLD}_{str_binary}.csv`

`str_choice`: During annotation, annotators could optionally give a second vote if they were uncertain about the assignment of the class. If this parameter is `second` the second vote was also considered. Otherwise this parameter is `first`. 

`THRESHOLD`: Specifies how many votes are needed for a class to be included in the dataset. If this parameter is 2, 2 of the 3 annotators must have assigned the same class. If the value is 3, all annotators must agree.

`str_binary`: This parameter can be either `bloom` or `binary`. If this value is `bloom` the Bloom taxonomy was used as classes.If this value is `binary` the Bloom classes were simplified and converted into binary classes (Easy and Heavy). 