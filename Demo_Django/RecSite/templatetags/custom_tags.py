from django import template

register = template.Library()

@register.filter(name='range_n')
def range_n(n):
    return range(n)


