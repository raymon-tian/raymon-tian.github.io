---
layout:     post
title:      python import system
subtitle:   import system
date:       2020-03-31
author:     DT
header-img: img/post-bg-debug.png
catalog: true
tags:
    - python



---

# 官方文档随记

1. python的import机制有多种：The [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement is the most common way of invoking the import machinery, but it is not the only way. Functions such as [`importlib.import_module()`](https://docs.python.org/3/library/importlib.html#importlib.import_module) and built-in [`__import__()`](https://docs.python.org/3/library/functions.html#__import__) can also be used to invoke the import machinery.

2. The [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement combines two operations; it searches for the named module, then it binds the results of that search to a name in the local scope. The search operation of the `import` statement is defined as a call to the [`__import__()`](https://docs.python.org/3/library/functions.html#__import__) function, with the appropriate arguments. The return value of [`__import__()`](https://docs.python.org/3/library/functions.html#__import__) is used to perform the name binding operation of the `import` statement.

3. When a module is first imported, Python searches for the module and if found, it creates a module object [1](https://docs.python.org/3/reference/import.html#fnmo), initializing it. If the named module cannot be found, a [`ModuleNotFoundError`](https://docs.python.org/3/library/exceptions.html#ModuleNotFoundError) is raised.

4. python中[module](https://docs.python.org/3/glossary.html#term-module)的定义：An object that serves as an organizational unit of Python code. Modules have a namespace containing arbitrary Python objects. Modules are loaded into Python by the process of [importing](https://docs.python.org/3/glossary.html#term-importing).

5. python中[package](https://docs.python.org/3/glossary.html#term-package)的定义：A Python [module](https://docs.python.org/3/glossary.html#term-module) which can contain submodules or recursively, subpackages. Technically, a package is a Python module with an `__path__` attribute.

6. python只有一种类型的module object：Python has only one type of module object, and all modules are of this type, regardless of whether the module is implemented in Python, C, or something else.

7. module与package的关系：To help organize modules and provide a naming hierarchy, Python has a concept of [packages](https://docs.python.org/3/glossary.html#term-package).

8. package是module，但是module并不一定是package；可以这么说：package是一种特殊的module；任何含有`__path__`属性的module就是package：It’s important to keep in mind that all packages are modules, but not all modules are packages. Or put another way, packages are just a special kind of module. Specifically, any module that contains a `__path__` attribute is considered a package.

9. All modules have a name. Subpackage names are separated from their parent package name by dots, akin to Python’s standard attribute access syntax.

10. **regular package**：A traditional [package](https://docs.python.org/3/glossary.html#term-package), such as a directory containing an `__init__.py` file.

11. **namespace package**：A [**PEP 420**](https://www.python.org/dev/peps/pep-0420) [package](https://docs.python.org/3/glossary.html#term-package) which serves only as a container for subpackages. Namespace packages may have no physical representation, and specifically are not like a [regular package](https://docs.python.org/3/glossary.html#term-regular-package) because they have no `__init__.py` file.

12. Python defines two types of packages, [regular packages](https://docs.python.org/3/glossary.html#term-regular-package) and [namespace packages](https://docs.python.org/3/glossary.html#term-namespace-package). Regular packages are traditional packages as they existed in Python 3.2 and earlier. **A regular package is typically implemented as a directory containing an `__init__.py` file. When a regular package is imported, this `__init__.py` file is implicitly executed**, and the objects it defines are bound to names in the package’s namespace. The `__init__.py` file can contain the same Python code that any other module can contain, and Python will add some additional attributes to the module when it is imported.

    For example, the following file system layout defines a top level `parent` package with three subpackages:

    ```python
    parent/
        __init__.py
        one/
            __init__.py
        two/
            __init__.py
        three/
            __init__.py
    ```

Importing `parent.one` will implicitly execute `parent/__init__.py` and `parent/one/__init__.py`. Subsequent imports of `parent.two` or `parent.three` will execute `parent/two/__init__.py` and `parent/three/__init__.py` respectively.（注意parent.\_\_init\_\_.py只会被调用一次）

13. **portion**：A set of files in a single directory (possibly stored in a zip file) that contribute to a namespace package, as defined in [**PEP 420**](https://www.python.org/dev/peps/pep-0420).
14. [namespace package](https://docs.python.org/3/reference/import.html#namespace-packages)：（比较抽象，跳过）

# reference

* https://docs.python.org/3/reference/import.html#importsystem