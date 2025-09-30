package main;
import java.io.FileNotFoundException;

public class CommentRemover {

    private StringBuffer fileContent;
    
    public void Run(StringBuffer buf) {
        fileContent = buf;
        

        RemoveComments();
        

        RemoveEmptyLines();
    }
 
    
    private boolean bInString = false;
    private boolean bInChar = false;
    private boolean bInSingleLineComment = false;
    private boolean bInMultiLineComment = false;

    char actChar = 0;
    char lastChar = 0;
    
    int actPos = 0;
    int deleteBegin = 0;
    
    private boolean InAnything()
    {
        return bInString || bInChar || bInSingleLineComment || bInMultiLineComment;
    }
    
    
    private void OnSlash()
    {

        if (bInMultiLineComment && lastChar=='*')
        {
            bInMultiLineComment = false;
            

            actPos++;
            
            Delete();


        actChar = 0;
        }

        else if (lastChar=='/' && !InAnything())
        {
            bInSingleLineComment = true;

            deleteBegin = actPos-1; 
        }
    }
    

    private void OnStar()
    {
        if (lastChar=='/' && !InAnything())
        {
            bInMultiLineComment = true;

            deleteBegin = actPos-1; 
        }
    }
    

    private void OnEOL()
    {
        if (bInSingleLineComment)
        {
            bInSingleLineComment = false;
            

            if (lastChar == '\r')
                actPos--;
            Delete();
        }
    }
    

    private void OnDoubleQuote()
    {

        if (bInString)
        {

            if (lastChar!='\\')
                bInString = false;
        }
        else if (!InAnything())
        {
            bInString = true;
        }
    }
    

    private void OnQuote()
    {

        if (bInChar)
        {

            if (lastChar!='\\')
                bInChar = false;
        }
        else if (!InAnything())
        {
            bInChar = true;
        }
    }
    

    private void Delete()
    {
        fileContent.delete(deleteBegin, actPos);
        

        actPos = deleteBegin + 1;
        

        lastChar = 0;
    } 
    
    private void RemoveComments()
    {
        actPos = -1;
        deleteBegin = 0;
        
        bInString = false;
        bInChar = false;
        bInSingleLineComment = false;
        bInMultiLineComment = false;

        actChar = 0;
        lastChar = 0;
        

        while (actPos < fileContent.length()-1)
        {
            actPos++;
            lastChar = actChar;
            actChar = fileContent.charAt(actPos);
            
            switch (actChar)
            {
            case '/' : OnSlash();break;
            case '\n': OnEOL();break;
            case '"' : OnDoubleQuote();break;
            case '\'': OnQuote();break;
            case '*': OnStar();break;
            }
        }
    }
        

    private void RemoveEmptyLines()
    {
        int start = 0;
        int end = 0;
        
        while (end!=-1)
        {
            start = fileContent.indexOf("\n",start);
            end = fileContent.indexOf("\n",start+1);
            if (start!=-1 && end!=-1)
            {
                String mid = fileContent.substring(start, end);
                mid = mid.trim();
                if (mid.length()==0)
                {
                    fileContent.delete(start, end);
                    start = 0;
                }
            }

            if (start >= 0)
                start++;
        }
        

        if (fileContent.substring(0, 1).equals("\n"))
            fileContent.delete(0, 1);
        else if (fileContent.substring(0, 2).equals("\r\n"))
            fileContent.delete(0, 2);
    }
}
