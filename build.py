# -*- coding: utf-8 -*-
# This file is part of the hdnet package
# Copyright 2014 the authors, see file AUTHORS.
# Licensed under the GPLv3, see file LICENSE for details

# pybuilder script

from pybuilder.core import init, task, description, depends, use_plugin

# set pythonpath (for unittests)
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# plugins
use_plugin("python.core")
use_plugin("python.pylint")
#use_plugin("python.flake8")
use_plugin("python.unittest")
#use_plugin("python.coverage")
use_plugin("python.distutils")
use_plugin("python.install_dependencies")
use_plugin("python.sphinx")
use_plugin("source_distribution")
use_plugin("copy_resources")
use_plugin("filter_resources")

# global config
name = "hdnet"
default_task = "release"

@init
def initialize(project):
    project.version = '1.0dev'

    # scripts path in egg
    #project.set_property("dir_dist_scripts", 'scripts')

    # build requirements
    #project.build_depends_on_requirements("requirements-dev.txt")

    # dist requirements
    #project.depends_on_requirements("requirements.txt")

    # core python
    project.set_property('dir_source_main_python', 'hdnet')
    project.set_property('dir_source_main_scripts', 'internal/scripts')
    #project.set_property('dir_dist', '$dir_target/dist/$name-$version')

    # pylint
    #project.set_property('pylint_options', ["--max-line-length=100", "--no-docstring-rgx=.*"])

    # flake8
    #project.set_property('flake8_ignore', "F403,W404,W801")
    project.set_property('flake8_include_test_sources', True)
    #project.set_property('flake8_exclude_patterns', '.svn,CVS,.bzr,.hg,.git,__pycache__')

    # unit tests
    project.set_property('dir_source_unittest_python', 'tests')
    project.set_property('unittest_module_glob', 'test_*')

    # coverage
    #project.set_property('coverage_break_build', False)

    # sphinx
    project.set_property('sphinx_config_path', 'doc')
    project.set_property('sphinx_source_dir', 'doc')
    project.set_property('sphinx_output_dir', 'doc/_build')

    # copy resources (non source files)
    project.set_property('copy_resources_glob', ['doc/_build/html/*', 'README.md'])

    # filter resources (placeholder replacement)
    # {variable} replaced by project.variable
    project.set_property('filter_resources_glob', ['doc/_build/html/*.html', 'README.md'])


@task
@description('Release new version of hdnet')
@depends('prepare', 'sphinx_generate_documentation', 'publish')
def release(project, logger):
    logger.info("Greetings master. I successfully built {0} in version {1}!".format(project.name, project.version))
