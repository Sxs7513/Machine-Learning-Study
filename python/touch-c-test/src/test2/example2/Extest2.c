#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int fac(int n){
	if(n<2)
		return 1;
	return n*fac(n-1);
}

char *reverse(char *s){
	register char t,
				*p = s,
				*q = s+strlen(s)-1;
	while(p<q){
		t = *p;
		*p++ = *q;
		*q-- = t;
	}
	return s;
}

int test()
{
	char s[BUFSIZ];

	printf("4! == %d\n",fac(4));
	printf("8! == %d\n",fac(8));
	printf("16! == %d\n",fac(16));

	printf("%d\n", BUFSIZ);

	strcpy(s,"abcdef");
	printf("%s\n",reverse(s));

	return 0;
}



#include "Python.h"

static PyObject *
Extest_fac(PyObject *self, PyObject* args){
	int num;
	if(!PyArg_ParseTuple(args,"i",&num))
		return NULL;
	return (PyObject*)Py_BuildValue("i",fac(num)); 
}


static PyObject *
Extest_doppel(PyObject*self,PyObject* args){
	char *orig_str;
	char *dupe_str;
	PyObject* retval;

	if(!PyArg_ParseTuple(args,"s",&orig_str))
		return NULL;
	retval = (PyObject*)Py_BuildValue("ss",orig_str,dupe_str=reverse(strdup(orig_str)));
	free(dupe_str);
	return retval;
}



static PyObject *
Extest_test(PyObject *self, PyObject* args){
	test();
	return (PyObject*)Py_BuildValue("");
}

static PyMethodDef
ExtestMethods[] = {
	{"fac",Extest_fac,METH_VARARGS},
	{"doppel",Extest_doppel,METH_VARARGS},
	{"test",Extest_test,METH_VARARGS},
	{NULL,NULL},
};

static struct PyModuleDef Extestmodule =
{
    PyModuleDef_HEAD_INIT,
    "Extest", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    ExtestMethods
};

PyMODINIT_FUNC PyInit_Extest(void)
{
    return PyModule_Create(&Extestmodule);
}