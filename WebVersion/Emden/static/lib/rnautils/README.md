# RNAutils

This package is deprecated and will not be updated!

However, development is continued here: https://github.com/ViennaRNA/fornac


### ColorScheme

Parse a string containing coloring information and store it in an object.

Example:

```javascript
let cs = new ColorScheme('10:red');
```

The important parts of the returned object will
look like this:

```javascript
{ 
  colorsText: '10:red',
  colorsJson: { '': { '10': 'red' } },
  range: [ 'white', 'steelblue' ],
  domain: [ Infinity, -Infinity ] }
}
```

Another exmaple using continuous colors:

```javascript
let cs = new ColorScheme('1:0.7 2:0.8');
```

Yields...

```javascript
{ 
  colorsText: '1:0.7 2:0.8',
  colorsJson: { '': { '1': 0.7, '2': 0.8 } },
  range: [ 'white', 'steelblue' ],
  domain: [ 0.7, 0.8] }
}
```

Apply colors only to molecules with a particular name:

```javascript
let cs = new ColorScheme('>blah\n 7:orange');
```
...

```javascript
{ 
  colorsText: '>blah\n 7:orange',
  colorsJson: { '': {} 'blah': { '7': orange} },
  range: [ 'white', 'steelblue' ],
  domain: [ Infinity, -Infinity] }
}
```

Specify a domain and a range for continuous values:

```javascript
let cs = new ColorScheme('domain=0:10 range=red:green 1: 0.7 2:0.8');
```
...

```javascript
{ 
  colorsText: '1:0.7 2:0.8',
  colorsJson: { '': { '1': 0.7, '2': 0.8 }},
  range: [ 'red', 'green' ],
  domain: [ '0', '10'] }
}
```

Other examples:

```javascript
let cs = new ColorScheme('1-3:red 4,5:blue');
```

## Development

First:

```
npm install
bower install
```

To debug:

```
gulp serve
```

To build:

```
gulp build
```

The output will be placed in the `dist` directory. To use `rnautils` in a web page, simply include `dist/scripts/rnautils.js` in your web page.
